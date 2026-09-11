#!/usr/bin/env bash
# ==============================================================================
# SCRIPT 1 BƯỚC DUY NHẤT CHẠY BENCHMARK TRÊN GOOGLE COLAB
# Sử dụng: bash run_colab.sh [--dataset all|uav123|anti_uav] [--models all|sglatrack|lgtrack] [--max_seqs N]
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=================================================================="
echo "          KIỂM TRA HỆ THỐNG & CẤU HÌNH GOOGLE COLAB"
echo "=================================================================="

# 1. Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[*] GPU phát hiện:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[!] Cảnh báo: Không phát hiện GPU NVIDIA. Đang dùng CPU."
fi

# 2. Cài đặt thư viện phụ thuộc
echo ""
echo "[1/4] Đang cài đặt thư viện phụ thuộc..."
pip install -q -r requirements.txt

# 3. Kiểm tra và chuẩn bị trọng số mô hình
echo ""
echo "[2/4] Kiểm tra checkpoints trọng số..."
bash checkpoints/download_weights.sh

# 4. Kiểm tra dữ liệu
echo ""
echo "[3/4] Tìm kiếm tập dữ liệu..."
DATA_DIR=""
POSSIBLE_DIRS=(
    "/content/drive/MyDrive/datasets"
    "/content/drive/MyDrive"
    "/content/datasets"
    "/content/data"
    "/home/nvidia/datasets"
    "$DIR/data"
)

for d in "${POSSIBLE_DIRS[@]}"; do
    if [ -d "$d/UAV123" ] || [ -d "$d/UAV-Anti-UAV" ]; then
        DATA_DIR="$d"
        echo "[+] Đã tìm thấy dataset tại: $DATA_DIR"
        break
    fi
done

if [ -z "$DATA_DIR" ]; then
    echo "[!] Chưa thấy dữ liệu sẵn có trong các thư mục thông thường."
    echo "[*] Gợi ý trên Colab: Bạn hãy mount Google Drive hoặc đặt dataset vào thư mục 'data/'"
fi

# 5. Chạy benchmark đánh giá (CHỈ XUẤT BOUNDING BOX VÀ CHỈ SỐ, KHÔNG XUẤT VIDEO)
echo ""
echo "[4/4] Bắt đầu chạy benchmark đánh giá..."
python3 evaluate.py --data_dir "$DATA_DIR" "$@"

echo ""
echo "=================================================================="
echo "          HOÀN THÀNH TẤT CẢ CÁC BƯỚC ĐÁNH GIÁ!"
echo " Báo cáo kết quả được lưu tại: $DIR/results/"
echo "=================================================================="
