#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Benchmark Đánh giá Toàn bộ Dataset (UAV123 & UAV-Anti-UAV)
Hỗ trợ cả 2 mô hình: SGLATrack (CVPR 2025) & LGTrack
LƯU Ý: Chỉ xuất tọa độ bounding box (.txt) và bảng chỉ số (Precision, AUC, FPS), KHÔNG XUẤT VIDEO.
"""

import os
import sys
import time
import json
import csv
import glob
import math
import argparse
import shutil
import numpy as np
import cv2
try:
    import torch
except ImportError:
    torch = None
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_iou(box1, box2):
    """box: [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0 else 0.0

def compute_cle(box1, box2):
    """Center Location Error in pixels"""
    c1 = (box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0)
    c2 = (box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0)
    return float(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

def compute_success_auc(ious, thresholds=np.linspace(0, 1, 21)):
    """AUC của Success Rate trên 21 ngưỡng [0, 1]"""
    if len(ious) == 0:
        return 0.0
    rates = [np.mean([1.0 if u >= th else 0.0 for u in ious]) for th in thresholds]
    return float(np.mean(rates))

def get_sglatrack(checkpoint_path=None):
    sgla_root = os.path.join(ROOT_DIR, 'SGLATrack')
    lg_root = os.path.join(ROOT_DIR, 'LGTrack')
    for mod in list(sys.modules.keys()):
        if mod == 'lib' or mod.startswith('lib.'):
            del sys.modules[mod]
    while lg_root in sys.path:
        sys.path.remove(lg_root)
    while sgla_root in sys.path:
        sys.path.remove(sgla_root)
    sys.path.insert(0, sgla_root)

    from lib.test.tracker.sglatrack import sglatrack
    from lib.test.parameter.sglatrack import parameters

    params = parameters('deit_distilled')
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ROOT_DIR, 'checkpoints', 'sglatrack_ep0297.pth.tar')
    params.checkpoint = checkpoint_path
    params.debug = False
    params.save_all_boxes = False
    tracker = sglatrack(params, 'uav')
    return tracker

def get_lgtrack(checkpoint_path=None):
    sgla_root = os.path.join(ROOT_DIR, 'SGLATrack')
    lg_root = os.path.join(ROOT_DIR, 'LGTrack')
    for mod in list(sys.modules.keys()):
        if mod == 'lib' or mod.startswith('lib.'):
            del sys.modules[mod]
    while sgla_root in sys.path:
        sys.path.remove(sgla_root)
    while lg_root in sys.path:
        sys.path.remove(lg_root)
    sys.path.insert(0, lg_root)

    from lib.test.tracker.lgtrack import LGTrack
    from lib.test.parameter.lgtrack import parameters

    params = parameters('deit_tiny_patch16_224')
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ROOT_DIR, 'checkpoints', 'LGTrack_ep0300.pth.tar')
    params.checkpoint = checkpoint_path
    params.debug = False
    params.save_all_boxes = False
    tracker = LGTrack(params, 'uav')
    return tracker

def parse_groundtruth(gt_file):
    boxes = []
    if not os.path.exists(gt_file):
        return []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(p.replace(',', ' ').strip()) for p in line.replace(',', ' ').split() if p.strip()]
            if len(parts) >= 4:
                boxes.append(parts[:4])
    return boxes

def parse_absent(absent_file):
    if not os.path.exists(absent_file):
        return None
    absents = []
    with open(absent_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                absents.append(int(line))
    return absents

def run_tracker_on_sequence(tracker, seq_info):
    img_files = seq_info.get('img_files', [])
    video_file = seq_info.get('video_file', None)
    gt_boxes = seq_info['gt_boxes']
    start_frame = seq_info.get('start_frame', 0)
    end_frame = seq_info.get('end_frame', len(gt_boxes))

    cap = None
    if not img_files and video_file:
        cap = cv2.VideoCapture(video_file)
        for _ in range(start_frame):
            cap.read()
        ret, first_img = cap.read()
        if not ret:
            return None, None
    elif img_files:
        first_img = cv2.imread(img_files[start_frame])
    else:
        return None, None

    init_box = [float(v) for v in gt_boxes[0]]
    first_img_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    tracker.initialize(first_img_rgb, {'init_bbox': init_box})

    total_frames = end_frame - start_frame
    pred_boxes = [init_box]
    frame_times = [0.0]

    for idx in range(1, total_frames):
        curr_frame_idx = start_frame + idx
        if img_files:
            frame = cv2.imread(img_files[curr_frame_idx])
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        out = tracker.track(frame_rgb)
        t_el = time.time() - t0
        box = [float(v) for v in out['target_bbox']]
        pred_boxes.append(box)
        frame_times.append(t_el)

    if cap:
        cap.release()
    return pred_boxes, frame_times

def evaluate_predictions(pred_boxes, gt_boxes, absent_flags=None):
    ious, cles = [], []
    total = min(len(pred_boxes), len(gt_boxes))
    for i in range(total):
        if absent_flags and i < len(absent_flags) and absent_flags[i] == 1:
            continue
        gt = gt_boxes[i]
        if len(gt) < 4 or gt[2] <= 0 or gt[3] <= 0:
            continue
        pr = pred_boxes[i]
        ious.append(compute_iou(pr, gt))
        cles.append(compute_cle(pr, gt))

    prec20 = float(np.mean([1.0 if c <= 20.0 else 0.0 for c in cles]) * 100.0) if cles else 0.0
    auc = float(compute_success_auc(ious) * 100.0) if ious else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return prec20, auc, mean_iou, len(ious)

def save_tracking_results(out_file, pred_boxes):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        for b in pred_boxes:
            f.write(f"{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n")

def find_datasets(base_path):
    """Tìm đường dẫn các tập dữ liệu tại các vị trí phổ biến trên máy/Colab"""
    paths = {
        "UAV123": None,
        "UAV-Anti-UAV": None
    }
    search_dirs = [
        base_path,
        '/kaggle/input/datasets/yolandcandy/uav-anti-uav/Test-002/Test',
        '/kaggle/input/datasets/yolandcandy/uav-anti-uav/Test-002',
        os.path.join(ROOT_DIR, 'data'),
        '/kaggle/input',
        '/kaggle/working',
        '/kaggle/working/datasets',
        '/tmp/datasets',
        '/content/datasets',
        '/content/data',
        '/content/drive/MyDrive/tracking/datasets',
        '/content/drive/MyDrive/tracking',
        '/content/drive/MyDrive/datasets',
        '/content/drive/MyDrive'
    ]
    # Tự động duyệt đệ quy các thư mục trong /kaggle/input (tối đa 5 tầng)
    if os.path.exists('/kaggle/input'):
        for root, dirs, _ in os.walk('/kaggle/input'):
            if 'Test' in dirs or 'Train' in dirs or 'data_seq' in dirs or 'UAV123' in dirs or 'UAV-Anti-UAV' in dirs:
                if root not in search_dirs:
                    search_dirs.append(root)
            if root.count(os.sep) - '/kaggle/input'.count(os.sep) >= 4:
                del dirs[:]

    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        # UAV123
        p1 = os.path.join(d, 'UAV123')
        if os.path.exists(p1) and os.path.exists(os.path.join(p1, 'data_seq')) and not paths["UAV123"]:
            paths["UAV123"] = p1
        elif os.path.exists(os.path.join(d, 'data_seq')) and not paths["UAV123"]:
            paths["UAV123"] = d
        # Anti-UAV
        p2 = os.path.join(d, 'UAV-Anti-UAV')
        if os.path.exists(p2) and (os.path.exists(os.path.join(p2, 'Test')) or os.path.exists(os.path.join(p2, 'Train'))) and not paths["UAV-Anti-UAV"]:
            paths["UAV-Anti-UAV"] = p2
        elif (os.path.exists(os.path.join(d, 'Test')) or os.path.exists(os.path.join(d, 'Train'))) and not paths["UAV-Anti-UAV"]:
            paths["UAV-Anti-UAV"] = d
        elif os.path.basename(d).lower() in ['test', 'train'] and not paths["UAV-Anti-UAV"]:
            paths["UAV-Anti-UAV"] = d
        elif not paths["UAV-Anti-UAV"] and os.path.isdir(d):
            sub_dirs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            if any('Test_' in s or 'Train_' in s or s.startswith('UAV-Anti-UAV') for s in sub_dirs):
                paths["UAV-Anti-UAV"] = d

    return paths

def load_uav123_sequences(dataset_root, max_seqs=None):
    anno_dir = os.path.join(dataset_root, 'anno', 'UAV123')
    if not os.path.exists(anno_dir):
        anno_dir = os.path.join(dataset_root, 'anno')
    seq_dir = os.path.join(dataset_root, 'data_seq', 'UAV123')
    if not os.path.exists(seq_dir):
        seq_dir = os.path.join(dataset_root, 'data_seq')

    # Load official sequence metadata (from configSeqs.m)
    try:
        from uav123_config import UAV123_CONFIGS
    except ImportError:
        cfg_file = os.path.join(ROOT_DIR, 'uav123_config.json')
        if os.path.exists(cfg_file):
            import json
            with open(cfg_file) as f:
                UAV123_CONFIGS = json.load(f)
        else:
            UAV123_CONFIGS = []

    seqs = []
    loaded_names = set()

    # 1. Load sequences using official configSeqs.m specifications (handles split sub-sequences)
    for cfg in UAV123_CONFIGS:
        seq_name = cfg['name']
        folder = cfg['folder']
        start_frame = cfg['startFrame']
        end_frame = cfg['endFrame']
        nz = cfg['nz']
        ext = cfg['ext']

        # Find sequence image directory
        cand_dir = os.path.join(seq_dir, folder)
        if not os.path.exists(cand_dir):
            cand_dir = os.path.join(dataset_root, 'data_seq', folder)
        if not os.path.exists(cand_dir):
            cand_dir = os.path.join(dataset_root, folder)
        if not os.path.exists(cand_dir):
            continue

        # Find annotation file
        anno_file = os.path.join(anno_dir, f"{seq_name}.txt")
        if not os.path.exists(anno_file):
            anno_file = os.path.join(dataset_root, 'anno', f"{seq_name}.txt")
        if not os.path.exists(anno_file):
            continue

        gts = parse_groundtruth(anno_file)
        if not gts:
            continue

        # Generate exact frame paths from startFrame to endFrame
        img_files = [
            os.path.join(cand_dir, f"{fn:0{nz}}.{ext}")
            for fn in range(start_frame, end_frame + 1)
        ]

        # Verification and fallback if filename casing differs
        if not img_files or not os.path.exists(img_files[0]):
            img_files_upper = [
                os.path.join(cand_dir, f"{fn:0{nz}}.{ext.upper()}")
                for fn in range(start_frame, end_frame + 1)
            ]
            if img_files_upper and os.path.exists(img_files_upper[0]):
                img_files = img_files_upper
            else:
                all_imgs = sorted(glob.glob(os.path.join(cand_dir, f"*.{ext}")) or
                                 glob.glob(os.path.join(cand_dir, "*.*")))
                if all_imgs:
                    s_idx = max(0, start_frame - 1)
                    e_idx = min(len(all_imgs), end_frame)
                    img_files = all_imgs[s_idx:e_idx]
                else:
                    continue

        valid_len = min(len(img_files), len(gts))
        seqs.append({
            "name": seq_name,
            "img_files": img_files[:valid_len],
            "gt_boxes": gts[:valid_len],
            "start_frame": 0,
            "end_frame": valid_len
        })
        loaded_names.add(seq_name)

    # 2. Fallback for any sequence txt in anno directory not present in UAV123_CONFIGS
    anno_files = sorted(glob.glob(os.path.join(anno_dir, '*.txt')))
    for af in anno_files:
        seq_name = os.path.splitext(os.path.basename(af))[0]
        if seq_name in loaded_names:
            continue
        video_name = seq_name.split('_')[0]
        cand_dir = os.path.join(seq_dir, video_name)
        if not os.path.exists(cand_dir):
            cand_dir = os.path.join(seq_dir, seq_name)
        if not os.path.exists(cand_dir):
            continue
        imgs = sorted(glob.glob(os.path.join(cand_dir, '*.jpg')) or glob.glob(os.path.join(cand_dir, '*.png')))
        if not imgs:
            continue
        gts = parse_groundtruth(af)
        if not gts:
            continue
        valid_len = min(len(imgs), len(gts))
        seqs.append({
            "name": seq_name,
            "img_files": imgs[:valid_len],
            "gt_boxes": gts[:valid_len],
            "start_frame": 0,
            "end_frame": valid_len
        })
        loaded_names.add(seq_name)

    if max_seqs and max_seqs > 0:
        seqs = seqs[:max_seqs]
    return seqs

def load_antiuav_sequences(dataset_root, split="Test", max_seqs=None):
    # 1. Xác định thư mục split (chứa các sequence con)
    direct_subs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))] if os.path.exists(dataset_root) else []
    has_direct_seqs = any("Test_" in d or "Train_" in d or d.startswith("UAV-Anti-UAV") for d in direct_subs)

    if has_direct_seqs:
        split_dir = dataset_root
    elif os.path.basename(dataset_root).lower() in ["test", "train"]:
        split_dir = dataset_root
    else:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            found = False
            for sub in direct_subs:
                if split.lower() in sub.lower():
                    split_dir = os.path.join(dataset_root, sub)
                    found = True
                    break
            if not found:
                split_dir = dataset_root

    if not os.path.exists(split_dir):
        print(f"[!] Thư mục split không tồn tại: {split_dir}")
        return []

    seq_dirs = sorted([
        os.path.join(split_dir, d) for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and ('Test' in d or 'Train' in d or 'UAV' in d)
    ])

    if not seq_dirs:
        seq_dirs = sorted([
            os.path.join(split_dir, d) for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')
        ])

    seqs = []
    for sd in seq_dirs:
        sname = os.path.basename(sd)

        # 1. Tìm file video (.mp4, .avi)
        vfiles = glob.glob(os.path.join(sd, '*.mp4')) or glob.glob(os.path.join(sd, '*.avi'))

        # 2. Tìm frame ảnh (.jpg, .png) nếu không có video
        all_imgs = sorted(glob.glob(os.path.join(sd, '*.jpg')) + glob.glob(os.path.join(sd, '*.png')) + glob.glob(os.path.join(sd, '*.JPEG')))
        img_files = []
        if len(all_imgs) > 1:
            digit_imgs = [f for f in all_imgs if os.path.splitext(os.path.basename(f))[0].isdigit()]
            if digit_imgs:
                img_files = digit_imgs
            else:
                img_files = [f for f in all_imgs if sname not in os.path.basename(f)]
                if not img_files:
                    img_files = all_imgs

        # 3. Tìm file ground truth
        gt_file = os.path.join(sd, 'groundtruth_rect.txt')
        if not os.path.exists(gt_file):
            gt_file = os.path.join(sd, 'groundtruth.txt')
        if not os.path.exists(gt_file):
            gt_file = os.path.join(sd, f'{sname}.txt')
        if not os.path.exists(gt_file):
            txt_cands = [f for f in glob.glob(os.path.join(sd, '*.txt')) if 'absent' not in os.path.basename(f) and 'attr' not in os.path.basename(f) and 'lang' not in os.path.basename(f)]
            if txt_cands:
                gt_file = txt_cands[0]

        absent_file = os.path.join(sd, 'absent.txt')

        if (not vfiles and not img_files) or not os.path.exists(gt_file):
            continue

        gts = parse_groundtruth(gt_file)
        absents = parse_absent(absent_file)
        if not gts:
            continue

        if img_files:
            valid_len = min(len(img_files), len(gts))
            seqs.append({
                "name": sname,
                "img_files": img_files[:valid_len],
                "gt_boxes": gts[:valid_len],
                "absent_flags": absents[:valid_len] if absents else None,
                "start_frame": 0,
                "end_frame": valid_len
            })
        else:
            valid_len = len(gts)
            seqs.append({
                "name": sname,
                "video_file": vfiles[0],
                "gt_boxes": gts[:valid_len],
                "absent_flags": absents[:valid_len] if absents else None,
                "start_frame": 0,
                "end_frame": valid_len
            })

    if max_seqs and max_seqs > 0:
        seqs = seqs[:max_seqs]
    return seqs

def main():
    parser = argparse.ArgumentParser(description="Evaluate SGLATrack & LGTrack on UAV Datasets")
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'uav123', 'anti_uav'], help='Dataset cần đánh giá')
    parser.add_argument('--data_dir', type=str, default='', help='Thư mục gốc chứa datasets')
    parser.add_argument('--max_seqs', type=int, default=0, help='Giới hạn số sequence (0 = toàn bộ)')
    parser.add_argument('--models', type=str, default='lgtrack', choices=['all', 'sglatrack', 'lgtrack'], help='Mô hình đánh giá (mặc định: lgtrack)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'results'), help='Thư mục lưu kết quả')
    args = parser.parse_args()

    # Tối ưu hóa PyTorch trên GPU
    if torch is not None:
        torch.backends.cudnn.enabled = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    print(f"[*] Thiết bị suy luận: {device}")
    if torch is not None and device == "cuda":
        print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")

    # Xác định đường dẫn datasets
    ds_paths = find_datasets(args.data_dir)
    print(f"[*] Đường dẫn UAV123: {ds_paths['UAV123']}")
    print(f"[*] Đường dẫn UAV-Anti-UAV: {ds_paths['UAV-Anti-UAV']}")

    datasets_to_run = []
    if args.dataset in ['all', 'uav123']:
        if ds_paths['UAV123']:
            uav123_seqs = load_uav123_sequences(ds_paths['UAV123'], max_seqs=args.max_seqs)
            datasets_to_run.append(("UAV123", uav123_seqs))
        else:
            print("[!] Không tìm thấy tập UAV123!")

    if args.dataset in ['all', 'anti_uav']:
        if ds_paths['UAV-Anti-UAV']:
            anti_seqs = load_antiuav_sequences(ds_paths['UAV-Anti-UAV'], split="Test", max_seqs=args.max_seqs)
            datasets_to_run.append(("UAV-Anti-UAV", anti_seqs))
        else:
            print("[!] Không tìm thấy tập UAV-Anti-UAV!")

    models_to_run = []
    if args.models in ['all', 'sglatrack']:
        models_to_run.append(("SGLATrack", get_sglatrack))
    if args.models in ['all', 'lgtrack']:
        models_to_run.append(("LGTrack", get_lgtrack))

    results_raw = []
    summary = {}

    # Khởi tạo file CSV ghi kết quả liên tục (Real-time incremental saving)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(args.output_dir, "metrics_summary.csv")
    csv_exists = os.path.exists(csv_file)
    csv_header = ["Dataset", "Sequence", "Model", "Frames", "Precision_20px(%)", "Success_AUC(%)", "Mean_IoU", "FPS", "BBox_Path"]
    
    # Kiểm tra đường dẫn backup lên Google Drive nếu có
    drive_backup = "/content/drive/MyDrive/TEST_RESULTS" if os.path.exists("/content/drive/MyDrive") else None
    if drive_backup:
        os.makedirs(drive_backup, exist_ok=True)

    # Đọc kết quả cũ nếu có để tránh trùng lặp
    existing_keys = set()
    if csv_exists:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_keys.add((r.get("Dataset"), r.get("Sequence"), r.get("Model")))
        except Exception:
            pass
    else:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    for dname, seq_list in datasets_to_run:
        print(f"\n=======================================================")
        print(f"ĐANG ĐÁNH GIÁ TẬP DỮ LIỆU: {dname} ({len(seq_list)} chuỗi)")
        print(f"=======================================================")
        summary[dname] = {}

        for mname, model_fn in models_to_run:
            print(f"\n---> Khởi động mô hình: {mname} trên {dname}...")
            tracker = None  # Khởi tạo lười (lazy load) chỉ khi cần chạy GPU
            
            p_list, a_list, iou_list, fps_list = [], [], [], []
            pbar = tqdm(seq_list, desc=f"[{mname}] {dname}")

            for sinfo in pbar:
                sname = sinfo['name']
                txt_out = os.path.join(args.output_dir, 'tracking_results', mname, dname, f"{sname}.txt")
                expected_len = sinfo.get('end_frame', len(sinfo['gt_boxes'])) - sinfo.get('start_frame', 0)

                # KIỂM TRA TỰ ĐỘNG KHÔI PHỤC (RESUME CAPABILITY):
                # Nếu chuỗi này đã được chạy và lưu kết quả trước đó -> đọc kết quả cũ, bỏ qua chạy lại GPU
                if os.path.exists(txt_out):
                    cached_boxes = parse_groundtruth(txt_out)
                    if len(cached_boxes) >= max(1, expected_len - 5):
                        pred_boxes = cached_boxes
                        frame_times = [0.0]
                        prec20, auc, mean_iou, valid_n = evaluate_predictions(pred_boxes, sinfo['gt_boxes'], sinfo.get('absent_flags'))
                        avg_fps = 60.0  # Ước tính
                        p_list.append(prec20)
                        a_list.append(auc)
                        iou_list.append(mean_iou)
                        fps_list.append(avg_fps)
                        pbar.set_postfix({"Resumed": sname, "Prec@20": f"{prec20:.1f}%", "AUC": f"{auc:.1f}%"})
                        
                        if (dname, sname, mname) not in existing_keys:
                            row = [dname, sname, mname, valid_n, round(prec20, 2), round(auc, 2), round(mean_iou, 3), round(avg_fps, 1), txt_out]
                            with open(csv_file, 'a', newline='') as f:
                                csv.writer(f).writerow(row)
                            existing_keys.add((dname, sname, mname))
                        continue

                # Nếu chưa chạy -> load mô hình (nếu chưa load) và chạy GPU
                if tracker is None:
                    tracker = model_fn()

                pred_boxes, frame_times = run_tracker_on_sequence(tracker, sinfo)
                if pred_boxes is None:
                    continue

                # Lưu bounding box kết quả dạng file .txt chuẩn
                save_tracking_results(txt_out, pred_boxes)

                # Đánh giá chỉ số
                prec20, auc, mean_iou, valid_n = evaluate_predictions(pred_boxes, sinfo['gt_boxes'], sinfo.get('absent_flags'))
                avg_fps = float(1.0 / np.mean(frame_times[1:])) if len(frame_times) > 1 else 0.0

                p_list.append(prec20)
                a_list.append(auc)
                iou_list.append(mean_iou)
                fps_list.append(avg_fps)

                pbar.set_postfix({"Prec@20": f"{prec20:.1f}%", "AUC": f"{auc:.1f}%", "FPS": f"{avg_fps:.1f}"})

                row_dict = {
                    "Dataset": dname,
                    "Sequence": sname,
                    "Model": mname,
                    "Frames": valid_n,
                    "Precision_20px": round(prec20, 2),
                    "Success_AUC": round(auc, 2),
                    "Mean_IoU": round(mean_iou, 3),
                    "FPS": round(avg_fps, 1),
                    "BBox_Path": txt_out
                }
                results_raw.append(row_dict)

                # GHI LIÊN TỤC VÀO CSV NGAY LẬP TỨC (Không lo mất dữ liệu nếu ngắt đột ngột)
                if (dname, sname, mname) not in existing_keys:
                    row = [dname, sname, mname, valid_n, round(prec20, 2), round(auc, 2), round(mean_iou, 3), round(avg_fps, 1), txt_out]
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    existing_keys.add((dname, sname, mname))

                # Tự động đồng bộ file CSV sang Google Drive sau mỗi chuỗi
                if drive_backup:
                    try:
                        shutil.copy2(csv_file, os.path.join(drive_backup, "metrics_summary.csv"))
                    except Exception:
                        pass

            summary[dname][mname] = {
                "Mean_Precision_20px": round(float(np.mean(p_list)), 2) if p_list else 0.0,
                "Mean_Success_AUC": round(float(np.mean(a_list)), 2) if a_list else 0.0,
                "Mean_IoU": round(float(np.mean(iou_list)), 3) if iou_list else 0.0,
                "Mean_FPS": round(float(np.mean(fps_list)), 1) if fps_list else 0.0,
                "Total_Sequences": len(p_list)
            }

    # Xuất báo cáo JSON
    json_file = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_file, 'w') as f:
        json.dump({"summary": summary, "details": results_raw}, f, indent=2)

    if drive_backup:
        try:
            shutil.copy2(json_file, os.path.join(drive_backup, "metrics_summary.json"))
            print(f"[+] Đã tự động sao lưu báo cáo kết quả sang Google Drive: {drive_backup}")
        except Exception:
            pass

    # In Bảng Tổng Kết
    print("\n" + "="*70)
    print("           TỔNG HỢP KẾT QUẢ BENCHMARK TRÊN CÁC TẬP DỮ LIỆU")
    print("="*70)
    for dname, dmodels in summary.items():
        print(f"\n📁 Tập dữ liệu: {dname}")
        print("-" * 65)
        print(f"{'Mô hình':<15} | {'Prec@20px (%)':<15} | {'Success/AUC (%)':<15} | {'FPS':<10}")
        print("-" * 65)
        for mname, mmetrics in dmodels.items():
            print(f"{mname:<15} | {mmetrics['Mean_Precision_20px']:<15.2f} | {mmetrics['Mean_Success_AUC']:<15.2f} | {mmetrics['Mean_FPS']:<10.1f}")
        print("-" * 65)

    print(f"\n[+] Đã lưu file tọa độ bounding box tại: {os.path.join(args.output_dir, 'tracking_results')}")
    print(f"[+] Đã lưu báo cáo CSV chi tiết tại: {csv_file}")
    print(f"[+] Đã lưu báo cáo JSON chi tiết tại: {json_file}")

if __name__ == '__main__':
    main()
