# TEST-SGLATRACK-LGTRACK

Dự án đánh giá benchmark toàn diện hai mô hình theo dõi mục tiêu UAV tiên tiến:
1. **SGLATrack** (CVPR 2025: *Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking*)
2. **LGTrack** (*Layer-Guided UAV Tracking: Enhancing Efficiency and Occlusion Robustness*)

Được thiết kế để chạy trơn tru trên **Google Colab** hoặc server với **1 câu lệnh bash duy nhất**.

> [!NOTE]
> Dự án được tối ưu **chỉ xuất tọa độ bounding box (`.txt`) và chỉ số đánh giá (`Precision`, `Success Rate / AUC`, `Mean IoU`, `FPS`)**, **KHÔNG xuất video** để tiết kiệm tối đa dung lượng bộ nhớ và thời gian chạy.

---

## ⚡ Hướng dẫn chạy 1 Cell trên Google Colab

### Bước 1: Mở Google Colab (chọn Runtime GPU: T4 / V100 / A100)

### Bước 2: Tạo một Code Cell duy nhất và chạy:

```bash
# 1. Clone repository về Colab
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/TEST-SGLATRACK-LGTRACK.git
%cd TEST-SGLATRACK-LGTRACK

# 2. Mount Google Drive nếu dữ liệu lưu trên Google Drive
from google.colab import drive
import os
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# 3. Chạy toàn bộ benchmark (1 lệnh bash)
!bash run_colab.sh --dataset all --models all
```

---

## ⚙️ Các tùy chọn tham số chạy nâng cao

Bạn có thể truyền các tham số tùy chọn vào `run_colab.sh` hoặc `evaluate.py`:

```bash
# Chạy đánh giá toàn bộ cả 2 tập UAV123 và UAV-Anti-UAV:
!bash run_colab.sh --dataset all

# Chỉ chạy đánh giá tập UAV123:
!bash run_colab.sh --dataset uav123

# Chỉ chạy đánh giá tập UAV-Anti-UAV:
!bash run_colab.sh --dataset anti_uav

# Chỉ chạy riêng mô hình SGLATrack hoặc LGTrack:
!bash run_colab.sh --models sglatrack
!bash run_colab.sh --models lgtrack

# Chạy thử nghiệm nhanh trên N sequence đầu tiên:
!bash run_colab.sh --max_seqs 5

# Chỉ định đường dẫn thư mục chứa dataset thủ công:
!bash run_colab.sh --data_dir "/content/drive/MyDrive/datasets"
```

---

## 📁 Cấu trúc thư mục dữ liệu yêu cầu

Script sẽ tự động tìm kiếm dữ liệu tại:
- `/content/drive/MyDrive/datasets` hoặc `/content/drive/MyDrive`
- `./data/`
- `/content/datasets`

Định dạng thư mục chuẩn:
```
datasets/
├── UAV123/
│   ├── anno/
│   │   └── UAV123/          # Chứa các file .txt ground truth
│   └── data_seq/
│       └── UAV123/          # Chứa các thư mục chuỗi ảnh .jpg
│
└── UAV-Anti-UAV/
    └── Test/                # Chứa các thư mục video UAV-Anti-UAV_Test_xxxxxx
        ├── UAV-Anti-UAV_Test_000001/
        │   ├── UAV-Anti-UAV_Test_000001.mp4
        │   ├── groundtruth_rect.txt
        │   └── absent.txt
        └── ...
```

---

## 📊 Kết quả đầu ra

Sau khi chạy xong, toàn bộ kết quả được lưu tại thư mục `results/`:
- `results/tracking_results/<Model>/<Dataset>/<Sequence>.txt`: Tọa độ bounding box dự đoán của từng khung hình (`x,y,w,h`).
- `results/metrics_summary.csv`: Bảng tổng hợp chỉ số Precision, AUC, IoU, FPS của từng chuỗi cho cả 2 mô hình.
- `results/metrics_summary.json`: Báo cáo chi tiết và điểm số trung bình định dạng JSON.

---

## 🚀 Đẩy Repo này lên GitHub của bạn

Từ máy Jetson hoặc máy local:
```bash
cd /home/nvidia/minh/TEST-SGLATRACK-LGTRACK
git init
git add .
git commit -m "Initial commit of TEST-SGLATRACK-LGTRACK benchmark"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/TEST-SGLATRACK-LGTRACK.git
git push -u origin main
```
*(Nếu file checkpoint lớn hơn 100MB, bạn có thể kích hoạt Git LFS: `git lfs track "*.pth*" "*.tar"` hoặc dùng script `checkpoints/download_weights.sh`).*
