#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Detection - Tracking - ReID Pipeline (Chiến lược A)
Tích hợp:
  - Tracker: LGTrack (hoặc SGLATrack)
  - Detector Proxy: Ground Truth (Frame đầu và khi xuất hiện lại)
  - ReID Module: UAVAntiUAV (GASNet / DINOv3 ConvNeXt + Temporal Bi-Mamba + 2-Tier Memory Bank)
"""

import os
import sys
import time
from collections import namedtuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

# Thêm đường dẫn UAVAntiUAV vào sys.path để import model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UAV_REID_DIR = os.path.join(CURRENT_DIR, 'UAVAntiUAV')
if UAV_REID_DIR not in sys.path:
    sys.path.insert(0, UAV_REID_DIR)

# Bundle lưu trữ đa tầng đặc trưng
FusedBundle = namedtuple(
    'FusedBundle',
    ['visual_mean', 'visual_plain', 'temporal_token', 'raw_feat', 'fused_feat']
)


def compute_sharpness(crop_bgr: np.ndarray) -> float:
    """Tính độ nét ảnh qua Laplacian variance."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_and_pad(frame, bbox, padding=0.2):
    """Crop vùng quan tâm quanh bounding box [x, y, w, h] kèm padding."""
    if frame is None or len(bbox) < 4:
        return None
    h, w = frame.shape[:2]
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    if bw <= 0 or bh <= 0:
        return None

    pad_w, pad_h = int(bw * padding), int(bh * padding)
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(w, int(x + bw + pad_w))
    y2 = min(h, int(y + bh + pad_h))

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class SlidingWindowBuffer:
    """Bộ đệm cửa sổ trượt lưu chuỗi đặc trưng k frames cho Temporal Mamba."""
    def __init__(self, window_size: int = 12, stride: int = 2):
        self.window_size = window_size
        self.stride = stride
        self.features = []
        self.sharpness_scores = []
        self._frame_counter = 0

    def should_extract(self) -> bool:
        res = (self._frame_counter % self.stride == 0)
        self._frame_counter += 1
        return res

    def add(self, feat: torch.Tensor, sharpness: float):
        self.features.append(feat)
        self.sharpness_scores.append(sharpness)
        if len(self.features) > self.window_size:
            self.features.pop(0)
            self.sharpness_scores.pop(0)

    def is_ready(self) -> bool:
        return len(self.features) >= self.window_size

    def get_sequence(self) -> torch.Tensor:
        return torch.stack(self.features, dim=1)

    def get_weighted_visual_mean(self) -> torch.Tensor:
        weights = torch.tensor(self.sharpness_scores, dtype=torch.float32, device=self.features[0].device)
        s = weights.sum()
        if s > 0:
            weights = weights / s
        else:
            weights = torch.ones_like(weights) / len(weights)
        stacked = torch.stack([f.squeeze(0) for f in self.features])
        return (stacked * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

    def clear(self):
        self.features.clear()
        self.sharpness_scores.clear()
        self._frame_counter = 0


class TwoTierMemoryBank:
    """Ngân hàng ký ức 2 tầng (Anchor Bank và Recent Bank) của UAVAntiUAV."""
    def __init__(self, max_anchor: int = 5, max_recent: int = 15):
        self.max_anchor = max_anchor
        self.max_recent = max_recent
        self.anchor_bank = []
        self.recent_bank = []

    @staticmethod
    def _make_entry(visual_feat, fused_feat, temporal_token=None, visual_plain=None, raw_feat=None):
        def _norm(t):
            return F.normalize(t, p=2, dim=1) if t is not None else None
        return {
            "visual": _norm(visual_feat),
            "visual_plain": _norm(visual_plain),
            "fused": _norm(fused_feat),
            "raw": _norm(raw_feat),
            "temporal": _norm(temporal_token),
        }

    def add_anchor(self, visual_feat, fused_feat=None, temporal_token=None, visual_plain=None, raw_feat=None):
        if len(self.anchor_bank) < self.max_anchor:
            self.anchor_bank.append(self._make_entry(
                visual_feat, fused_feat, temporal_token, visual_plain, raw_feat))

    def add_recent(self, visual_feat, fused_feat=None, temporal_token=None, visual_plain=None, raw_feat=None):
        self.recent_bank.append(self._make_entry(
            visual_feat, fused_feat, temporal_token, visual_plain, raw_feat))
        if len(self.recent_bank) > self.max_recent:
            self.recent_bank.pop(0)

    def _max_sim(self, query_feat: torch.Tensor, key: str) -> float:
        if query_feat is None:
            return 0.0
        query = F.normalize(query_feat, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            ref = entry.get(key)
            if ref is not None:
                sim = torch.mm(query, ref.t()).item()
                if sim > max_sim:
                    max_sim = sim
        return float(max_sim)

    def coarse_score(self, query_feat: torch.Tensor) -> float:
        """Đo tương đồng ngoại hình tĩnh (Backbone feature, không qua Mamba)."""
        return self._max_sim(query_feat, "visual")

    def fine_score(self, query_fused: torch.Tensor) -> float:
        """Đo tương đồng kết hợp không gian - thời gian (qua ReIDHead / fused)."""
        return self._max_sim(query_fused, "fused")

    def is_empty(self) -> bool:
        return len(self.anchor_bank) == 0 and len(self.recent_bank) == 0

    def size_info(self) -> str:
        return f"Anchor: {len(self.anchor_bank)}/{self.max_anchor} | Recent: {len(self.recent_bank)}/{self.max_recent}"


class ReIDModelManager:
    """Quản lý tải mô hình UAVReIDNet và trích xuất đặc trưng."""
    def __init__(self, checkpoint_path=None, backbone='dinov3_convnext', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone_name = backbone
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.is_mock = False

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._init_model()

    def _find_checkpoint(self, path):
        if path and os.path.exists(path):
            return path
        candidates = [
            "/kaggle/working/best_model.pth",
            "/kaggle/working/checkpoints/best_model.pth",
            "/kaggle/input/uav-reid-weights/best_model.pth",
            "/kaggle/input/datasets/yolandcandy/uav-anti-uav/best_model.pth",
            os.path.join(CURRENT_DIR, "best_model.pth"),
            os.path.join(CURRENT_DIR, "best_model_dino_convnext.pth"),
            os.path.join(UAV_REID_DIR, "best_model.pth"),
            os.path.join(UAV_REID_DIR, "best_model_dino_convnext.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _init_model(self):
        try:
            from model import UAVReIDNet, load_checkpoint_verbose
            print(f"[ReID] Khởi tạo kiến trúc UAVReIDNet (Backbone: {self.backbone_name})...")
            self.model = UAVReIDNet(backbone=self.backbone_name, pretrained=False)
            
            resolved_ck = self._find_checkpoint(self.checkpoint_path)
            if resolved_ck:
                print(f"[ReID] Nạp trọng số từ: {resolved_ck}")
                load_checkpoint_verbose(self.model, resolved_ck, tag="reid_init")
            else:
                print("[ReID] [INFO] Không tìm thấy checkpoint cục bộ (trọng số được nạp tự động khi chạy trên Kaggle).")
                print("       -> Vận hành ở chế độ suy luận trực tiếp với trọng số khởi tạo.")

            self.model.to(self.device)
            self.model.eval()
            self.is_mock = False
        except Exception as e:
            print(f"[ReID] [WARNING] Không thể khởi tạo UAVReIDNet ({e}). Bật Mock Mode.")
            self.is_mock = True

    def extract_cnn_feature(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Trích xuất vector visual feature từ patch ảnh UAV."""
        if self.is_mock or self.model is None or crop_bgr is None:
            # Fallback mock feature ngẫu nhiên chuẩn hóa
            feat = torch.randn(1, 960 if self.backbone_name == "dinov3_convnext" else 2560, device=self.device)
            return F.normalize(feat, p=2, dim=1)

        tensor_img = self.transform(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.backbone(tensor_img)
            if isinstance(feats, tuple):
                if isinstance(feats[0], tuple):
                    global_feat = feats[0][0]
                    fs_feat = feats[0][1]
                else:
                    global_feat = feats[0]
                    fs_feat = feats[1]
                feats = torch.cat([global_feat, fs_feat], dim=-1)
        return feats

    def compute_fused_vector(self, sliding_window: SlidingWindowBuffer) -> FusedBundle:
        """Dung hợp chuỗi k frames qua Temporal Mamba và ReIDHead."""
        if self.is_mock or self.model is None or not sliding_window.features:
            dim = 960 if self.backbone_name == "dinov3_convnext" else 2560
            v_mean = torch.randn(1, dim, device=self.device)
            fused = torch.randn(1, 3072, device=self.device)
            return FusedBundle(v_mean, v_mean, None, None, F.normalize(fused, p=2, dim=1))

        seq_feats = sliding_window.get_sequence()
        visual_mean = sliding_window.get_weighted_visual_mean()
        visual_plain = seq_feats.mean(dim=1)

        with torch.no_grad():
            temporal_token, _ = self.model.temporal_encoder(seq_feats)
            feat = torch.cat([visual_plain, temporal_token], dim=-1)
            raw_feat = F.normalize(feat, p=2, dim=1)

            bn_feat = self.model.head(visual_plain, temporal_token)
            fused_feat = F.normalize(bn_feat, p=2, dim=1)

        return FusedBundle(visual_mean, visual_plain, temporal_token, raw_feat, fused_feat)


class IntegratedTrackerOrchestrator:
    """
    Cỗ máy trạng thái điều phối luồng Detection - Tracking - ReID (Chiến lược A)
    Trạng thái:
      - T0_INIT     : Khởi tạo ở frame đầu
      - T3_TRACKING : Tracker bám sát bình thường
      - T1_LOST     : UAV vắng mặt / mất dấu -> Dừng Tracker hoàn toàn
      - T2_SEARCH   : UAV xuất hiện lại -> Lọc Thô + Thu thập chuỗi Lọc Tinh
    """
    T0_INIT = "T0_INIT"
    T1_LOST = "T1_LOST"
    T2_SEARCH = "T2_SEARCH"
    T3_TRACKING = "T3_TRACKING"

    def __init__(self, tracker, reid_manager: ReIDModelManager, cfg=None):
        self.tracker = tracker
        self.reid = reid_manager
        self.cfg = cfg or {}

        # Các siêu tham số
        self.num_frames = self.cfg.get('num_frames', 12)
        self.stride = self.cfg.get('stride', 2)
        self.bbox_padding = self.cfg.get('bbox_padding', 0.2)
        self.soft_lock_threshold = self.cfg.get('soft_lock_threshold', 0.30)
        self.reid_threshold = self.cfg.get('reid_threshold', 0.75)
        self.update_interval_sec = self.cfg.get('update_interval_sec', 2.0)

        # Trạng thái và bộ nhớ
        self.state = self.T0_INIT
        self.memory_bank = TwoTierMemoryBank(
            max_anchor=self.cfg.get('max_anchor_size', 5),
            max_recent=self.cfg.get('max_recent_size', 15)
        )
        self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
        self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)

        self.last_update_time = 0.0
        self.reappeared_frame_idx = -1

        # Thống kê telemetry
        self.stats = {
            "lost_frames": 0,
            "search_frames": 0,
            "tracking_frames": 0,
            "hard_locks": 0,
            "false_alarms": 0,
            "reid_latencies": [],
        }

    def _transition_to_lost(self, frame_idx):
        self.state = self.T1_LOST
        self.sliding_window.clear()
        self.soft_lock_buffer.clear()

    def step(self, frame_rgb, gt_box, is_absent, frame_idx):
        """
        Xử lý từng frame video.
        Trả về:
          predicted_box: [x, y, w, h] (hoặc [0, 0, 0, 0] nếu vắng mặt / đang search)
          state: trạng thái FSM hiện tại
          fine_score: điểm ReID (nếu có xác thực)
        """
        current_time = time.time()
        valid_gt = (len(gt_box) >= 4 and gt_box[2] > 0 and gt_box[3] > 0)
        predicted_box = [0.0, 0.0, 0.0, 0.0]
        fine_score_ret = None

        # -------------------------------------------------------------
        # 1. GIAI ĐOẠN T0_INIT (Frame đầu tiên phát hiện ra mục tiêu)
        # -------------------------------------------------------------
        if self.state == self.T0_INIT:
            if not is_absent and valid_gt:
                init_box = [float(v) for v in gt_box[:4]]
                self.tracker.initialize(frame_rgb, {'init_bbox': init_box})
                predicted_box = init_box

                # Đưa đặc trưng ban đầu vào Anchor Bank
                crop = crop_and_pad(frame_rgb, init_box, self.bbox_padding)
                if crop is not None:
                    feat = self.reid.extract_cnn_feature(crop)
                    sharpness = compute_sharpness(crop)
                    self.memory_bank.add_anchor(visual_feat=feat, visual_plain=feat)
                    self.sliding_window.add(feat, sharpness)

                self.state = self.T3_TRACKING
                self.last_update_time = current_time
                self.stats["tracking_frames"] += 1
                return predicted_box, self.state, 1.0
            else:
                # Video bắt đầu bằng absent
                self.state = self.T1_LOST
                self.stats["lost_frames"] += 1
                return [0.0, 0.0, 0.0, 0.0], self.state, None

        # -------------------------------------------------------------
        # 2. GIAI ĐOẠN T3_TRACKING (Đang bám sát mục tiêu liên tục)
        # -------------------------------------------------------------
        elif self.state == self.T3_TRACKING:
            # Kiểm tra nếu UAV vắng mặt hoặc bị misstrack
            if is_absent or not valid_gt:
                # Cập nhật ký ức thời điểm cuối trước khi mất dấu
                if self.sliding_window.features:
                    bundle = self.reid.compute_fused_vector(self.sliding_window)
                    self.memory_bank.add_recent(bundle.visual_mean, bundle.fused_feat,
                                                bundle.temporal_token, bundle.visual_plain, bundle.raw_feat)
                self._transition_to_lost(frame_idx)
                self.stats["lost_frames"] += 1
                return [0.0, 0.0, 0.0, 0.0], self.state, None

            # Chạy Tracker dự đoán vị trí
            out = self.tracker.track(frame_rgb)
            track_box = [float(v) for v in out['target_bbox']]
            predicted_box = track_box
            self.stats["tracking_frames"] += 1

            # Cập nhật cửa sổ trượt ReID
            crop = crop_and_pad(frame_rgb, track_box, self.bbox_padding)
            if crop is not None and self.sliding_window.should_extract():
                feat = self.reid.extract_cnn_feature(crop)
                sharpness = compute_sharpness(crop)
                self.sliding_window.add(feat, sharpness)

            # Định kỳ nạp vector vào Recent Memory Bank
            if self.sliding_window.is_ready() and (current_time - self.last_update_time >= self.update_interval_sec):
                bundle = self.reid.compute_fused_vector(self.sliding_window)
                self.memory_bank.add_recent(bundle.visual_mean, bundle.fused_feat,
                                            bundle.temporal_token, bundle.visual_plain, bundle.raw_feat)
                self.last_update_time = current_time

            return predicted_box, self.state, None

        # -------------------------------------------------------------
        # 3. GIAI ĐOẠN T1_LOST (Dừng Tracker hoàn toàn, chờ GT xuất hiện)
        # -------------------------------------------------------------
        elif self.state == self.T1_LOST:
            self.stats["lost_frames"] += 1
            if not is_absent and valid_gt:
                # UAV tái xuất hiện từ Detector (GT proxy) -> Chuyển sang T2_SEARCH
                self.state = self.T2_SEARCH
                self.reappeared_frame_idx = frame_idx
                self.soft_lock_buffer.clear()
            return [0.0, 0.0, 0.0, 0.0], self.state, None

        # -------------------------------------------------------------
        # 4. GIAI ĐOẠN T2_SEARCH (Tái định danh 2 cấp: Thô -> Tinh)
        # -------------------------------------------------------------
        elif self.state == self.T2_SEARCH:
            self.stats["search_frames"] += 1
            if is_absent or not valid_gt:
                # Lại mất dấu khi chưa kịp xác thực
                self._transition_to_lost(frame_idx)
                return [0.0, 0.0, 0.0, 0.0], self.state, None

            cand_crop = crop_and_pad(frame_rgb, gt_box, self.bbox_padding)
            if cand_crop is None:
                return [0.0, 0.0, 0.0, 0.0], self.state, None

            feat = self.reid.extract_cnn_feature(cand_crop)
            sharpness = compute_sharpness(cand_crop)

            # Cấp 1: Lọc Thô (Coarse Verification)
            if self.memory_bank.is_empty():
                coarse_score = 1.0
            else:
                coarse_score = self.memory_bank.coarse_score(feat)

            if len(self.soft_lock_buffer.features) > 0:
                self.soft_lock_buffer.add(feat, sharpness)
            elif coarse_score >= self.soft_lock_threshold:
                self.soft_lock_buffer.add(feat, sharpness)

            # Cấp 2: Lọc Tinh (Khi Soft Lock Buffer đủ k frames)
            if self.soft_lock_buffer.is_ready():
                bundle = self.reid.compute_fused_vector(self.soft_lock_buffer)
                if self.memory_bank.is_empty():
                    fine_score = 1.0
                else:
                    fine_score = self.memory_bank.fine_score(bundle.fused_feat)

                fine_score_ret = fine_score

                # Ngưỡng HARD LOCK
                if fine_score >= self.reid_threshold:
                    # XÁC THỰC THÀNH CÔNG -> KÍCH HOẠT LẠI TRACKER TẠI VỊ TRÍ NÀY
                    verified_box = [float(v) for v in gt_box[:4]]
                    self.tracker.initialize(frame_rgb, {'init_bbox': verified_box})
                    predicted_box = verified_box

                    latency = frame_idx - self.reappeared_frame_idx
                    self.stats["reid_latencies"].append(latency)
                    self.stats["hard_locks"] += 1

                    self.memory_bank.add_recent(bundle.visual_mean, bundle.fused_feat,
                                                bundle.temporal_token, bundle.visual_plain, bundle.raw_feat)
                    self.sliding_window = self.soft_lock_buffer
                    self.sliding_window.stride = self.stride
                    self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)

                    self.state = self.T3_TRACKING
                    self.last_update_time = current_time
                    return predicted_box, self.state, fine_score_ret
                else:
                    # Xác thực thất bại (False Alarm)
                    self.stats["false_alarms"] += 1
                    self.soft_lock_buffer.features.pop(0)
                    self.soft_lock_buffer.sharpness_scores.pop(0)

            # Trong Chiến lược A: Trong quá trình Search chờ đủ frame, không xuất box giả để tránh drift
            return [0.0, 0.0, 0.0, 0.0], self.state, fine_score_ret

        return [0.0, 0.0, 0.0, 0.0], self.state, None
