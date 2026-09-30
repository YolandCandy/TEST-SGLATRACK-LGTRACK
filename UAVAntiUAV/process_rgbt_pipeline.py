import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from tqdm import tqdm

def crop_and_resize(frame, bbox, padding, crop_size):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    
    pad_w = bw * padding
    pad_h = bh * padding
    
    x1 = int(max(0, x - pad_w))
    y1 = int(max(0, y - pad_h))
    x2 = int(min(w, x + bw + pad_w))
    y2 = int(min(h, y + bh + pad_h))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = frame[y1:y2, x1:x2]
    try:
        resized = cv2.resize(crop, (crop_size, crop_size))
        return resized
    except Exception as e:
        return None

def process_sequence(seq_path, output_base, split, num_before_frames=16, num_after_frames=16, frame_stride=1, bbox_padding=0.2, crop_size=256):
    seq_name = os.path.basename(seq_path)
    
    video_path = os.path.join(seq_path, "visible.mp4")
    json_path = os.path.join(seq_path, "visible.json")
    
    if not os.path.exists(video_path) or not os.path.exists(json_path):
        return {"status": "error", "message": f"Missing files in {seq_path}"}
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        exist = data.get("exist", [])
        bboxes = data.get("gt_rect", [])
        
        absent = [1 if e == 0 else 0 for e in exist]
    except Exception as e:
        return {"status": "error", "message": f"Error reading json in {seq_path}: {e}"}

    events = []
    in_disappearance = False
    disappear_start = -1
    
    for i in range(len(absent)):
        if absent[i] == 1 and not in_disappearance:
            in_disappearance = True
            disappear_start = i
        elif absent[i] == 0 and in_disappearance:
            in_disappearance = False
            disappear_end = i
            events.append((disappear_start, disappear_end))
            
    if not events:
        return {"status": "skipped", "message": f"No disappearance in {seq_path}", "seq_name": seq_name}
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return {"status": "error", "message": f"Could not read video {video_path}"}
        
    pairs = []
    
    for event_idx, (start_idx, end_idx) in enumerate(events):
        # Gallery: before disappearance
        t1 = start_idx - 1
        before_sampled = [t1 - i * frame_stride for i in range(num_before_frames)]
        before_sampled = [f for f in before_sampled if f >= 0]
        before_sampled.sort()
        
        # Query: after reappearance
        t2 = end_idx
        after_sampled = [t2 + i * frame_stride for i in range(num_after_frames)]
        after_sampled = [f for f in after_sampled if f < total_frames]
        after_sampled.sort()
        
        needed_frames = sorted(list(set(before_sampled + after_sampled)))
        
        if not needed_frames:
            continue
            
        event_out_dir = os.path.join(output_base, split, f"{seq_name}_event_{event_idx}")
        before_dir = os.path.join(event_out_dir, "before")
        after_dir = os.path.join(event_out_dir, "after")
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        
        before_frames_files = []
        after_frames_files = []
        
        for frame_idx in needed_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            if frame_idx >= len(bboxes):
                continue
                
            bbox = bboxes[frame_idx]
            # [x, y, w, h] format
            if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
                
            crop = crop_and_resize(frame, bbox, bbox_padding, crop_size)
            if crop is None:
                continue
                
            frame_name = f"frame_{frame_idx:04d}.jpg"
            if frame_idx < start_idx:
                out_path = os.path.join(before_dir, frame_name)
                cv2.imwrite(out_path, crop)
                before_frames_files.append(frame_name)
            else:
                out_path = os.path.join(after_dir, frame_name)
                cv2.imwrite(out_path, crop)
                after_frames_files.append(frame_name)
                
        if before_frames_files and after_frames_files:
            pairs.append({
                "sequence_id": seq_name,
                "event_index": event_idx,
                "identity_id": None, # Will be assigned later globally
                "gallery_frames": before_frames_files,
                "query_frames": after_frames_files,
                "gallery_dir": f"{seq_name}_event_{event_idx}/before",
                "query_dir": f"{seq_name}_event_{event_idx}/after",
                "disappearance_duration_frames": end_idx - start_idx,
                "language_description": "",
                "attributes": []
            })
            
    cap.release()
    
    return {
        "status": "success",
        "seq_name": seq_name,
        "pairs": pairs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbt-data-dir", default="../Anti-UAV-RGBT")
    parser.add_argument("--processed-dir", default="../UAVAntiUAV/processed")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    import yaml
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if "data_pipeline" in cfg:
            dp = cfg["data_pipeline"]
            args.rgbt_data_dir = dp.get("anti_uav_rgbt_dir", args.rgbt_data_dir)
            args.processed_dir = dp.get("output_dir", args.processed_dir)
            args.frame_stride = dp.get("frame_stride", args.frame_stride)
    
    rgbt_data_dir = args.rgbt_data_dir
    processed_dir = args.processed_dir
    
    query_train_json = os.path.join(processed_dir, "query_train.json")
    gallery_train_json = os.path.join(processed_dir, "gallery_train.json")
    query_test_json = os.path.join(processed_dir, "query_test.json")
    gallery_test_json = os.path.join(processed_dir, "gallery_test.json")
    
    print(f"Loading existing JSON files...")
    with open(query_train_json, "r") as f:
        existing_query_train = json.load(f)
    with open(gallery_train_json, "r") as f:
        existing_gallery_train = json.load(f)
    with open(query_test_json, "r") as f:
        existing_query_test = json.load(f)
    with open(gallery_test_json, "r") as f:
        existing_gallery_test = json.load(f)
        
    all_existing_pids = [p["identity_id"] for p in existing_query_train + existing_query_test if p["identity_id"] is not None]
    max_id = max(all_existing_pids) if all_existing_pids else -1
    print(f"Current maximum identity_id: {max_id}")
    
    splits = {
        "train": ["train", "val"],
        "test": ["test"]
    }
    
    new_train_pairs = []
    new_test_pairs = []
    
    for out_split, rgbt_splits in splits.items():
        tasks = []
        for rgbt_split in rgbt_splits:
            split_dir = os.path.join(rgbt_data_dir, rgbt_split)
            if not os.path.exists(split_dir):
                continue
                
            seqs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
            for seq_name in seqs:
                seq_path = os.path.join(split_dir, seq_name)
                tasks.append((seq_path, processed_dir, out_split, 16, 16, args.frame_stride))
                
        out_list = new_train_pairs if out_split == "train" else new_test_pairs
        
        print(f"Start processing {len(tasks)} sequences for {out_split}...")
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_sequence, *task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Processing {out_split}"):
                res = future.result()
                if res["status"] == "success" and res["pairs"]:
                    out_list.extend(res["pairs"])
                    
    # Assign unique IDs
    unique_new_seqs = set()
    for p in new_train_pairs + new_test_pairs:
        unique_new_seqs.add(p["sequence_id"])
        
    seq_to_id = {seq: max_id + 1 + i for i, seq in enumerate(sorted(list(unique_new_seqs)))}
    
    for p in new_train_pairs:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        existing_gallery_train.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["gallery_frames"],
            "frame_dir": p["gallery_dir"]
        })
        existing_query_train.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["query_frames"],
            "frame_dir": p["query_dir"],
            "disappearance_duration_frames": p["disappearance_duration_frames"],
            "language_description": p["language_description"],
            "attributes": p["attributes"]
        })
        
    for p in new_test_pairs:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        existing_gallery_test.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["gallery_frames"],
            "frame_dir": p["gallery_dir"]
        })
        existing_query_test.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["query_frames"],
            "frame_dir": p["query_dir"],
            "disappearance_duration_frames": p["disappearance_duration_frames"],
            "language_description": p["language_description"],
            "attributes": p["attributes"]
        })
        
    print("Writing updated JSON files...")
    with open(query_train_json, "w") as f:
        json.dump(existing_query_train, f, indent=4)
    with open(gallery_train_json, "w") as f:
        json.dump(existing_gallery_train, f, indent=4)
        
    with open(query_test_json, "w") as f:
        json.dump(existing_query_test, f, indent=4)
    with open(gallery_test_json, "w") as f:
        json.dump(existing_gallery_test, f, indent=4)
        
    print(f"DONE! Added {len(new_train_pairs)} pairs to train and {len(new_test_pairs)} pairs to test.")

if __name__ == "__main__":
    main()
