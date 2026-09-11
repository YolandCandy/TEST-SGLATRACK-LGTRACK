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

# 4. Kiểm tra và chuẩn bị dữ liệu
echo ""
echo "[3/4] Tìm kiếm và chuẩn bị tập dữ liệu..."

# Tự động phát hiện và giải nén nếu dữ liệu nằm ở dạng file zip trên Drive sang ổ cứng Colab (/content/datasets)
LOCAL_DS="/content/datasets"
DRIVE_CANDIDATES=(
    "/content/drive/MyDrive/tracking/datasets"
    "/content/drive/MyDrive/tracking"
    "/content/drive/MyDrive/datasets"
    "/content/drive/MyDrive"
)

if [ -d "/content" ]; then
    mkdir -p "$LOCAL_DS"
    
    # 4.1 Kiểm tra UAV123
    if [ ! -d "$LOCAL_DS/UAV123" ]; then
        UAV_ZIP=""
        for d in "${DRIVE_CANDIDATES[@]}"; do
            for f in "$d/UAV123/UAV123.zip" "$d/UAV123.zip" "$d/UAV123/"*.zip "$d/"*UAV123*.zip; do
                if [ -f "$f" ]; then
                    UAV_ZIP="$f"
                    break 2
                fi
            done
        done
        if [ -n "$UAV_ZIP" ]; then
            echo "[+] Tìm thấy $UAV_ZIP. Đang copy sang Colab SSD và giải nén..."
            cp "$UAV_ZIP" "$LOCAL_DS/"
            ZIP_NAME=$(basename "$UAV_ZIP")
            mkdir -p "$LOCAL_DS/UAV123"
            unzip -q -o "$LOCAL_DS/$ZIP_NAME" -d "$LOCAL_DS/UAV123/"
            rm -f "$LOCAL_DS/$ZIP_NAME"
            echo "[+] Đã giải nén UAV123 thành công vào $LOCAL_DS/UAV123"
        fi
    fi

    # Tự động chuẩn hóa nếu data_seq nằm ở ngoài /content/datasets
    if [ -d "$LOCAL_DS/data_seq" ] && [ ! -d "$LOCAL_DS/UAV123/data_seq" ]; then
        mkdir -p "$LOCAL_DS/UAV123"
        mv "$LOCAL_DS/data_seq" "$LOCAL_DS/UAV123/" 2>/dev/null || true
        [ -d "$LOCAL_DS/anno" ] && mv "$LOCAL_DS/anno" "$LOCAL_DS/UAV123/" 2>/dev/null || true
    fi

    # 4.2 Kiểm tra UAV-Anti-UAV (Chỉ cần tập Test)
    if [ ! -d "$LOCAL_DS/UAV-Anti-UAV/Test" ]; then
        mkdir -p "$LOCAL_DS/UAV-Anti-UAV"
        ANTI_ZIP=""
        for d in "${DRIVE_CANDIDATES[@]}"; do
            for f in "$d/UAV-Anti-UAV/"Test*.zip "$d/UAV-Anti-UAV/"test*.zip "$d/"Test*.zip "$d/"test*.zip; do
                if [ -f "$f" ]; then
                    ANTI_ZIP="$f"
                    break 2
                fi
            done
        done
        if [ -n "$ANTI_ZIP" ]; then
            echo "[+] Tìm thấy $ANTI_ZIP. Đang copy sang Colab SSD và giải nén..."
            cp "$ANTI_ZIP" "$LOCAL_DS/"
            ZIP_NAME=$(basename "$ANTI_ZIP")
            unzip -q -o "$LOCAL_DS/$ZIP_NAME" -d "$LOCAL_DS/UAV-Anti-UAV/"
            rm -f "$LOCAL_DS/$ZIP_NAME"
            echo "[+] Đã giải nén UAV-Anti-UAV Test thành công vào $LOCAL_DS/UAV-Anti-UAV/Test"
        fi
    fi
fi

DATA_DIR=""
POSSIBLE_DIRS=(
    "/content/datasets"
    "/content/drive/MyDrive/tracking/datasets"
    "/content/drive/MyDrive/tracking"
    "/content/drive/MyDrive/datasets"
    "/content/drive/MyDrive"
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
    echo "[!] Cảnh báo: Chưa thấy thư mục dữ liệu đã giải nén."
    echo "[*] Vui lòng đảm bảo đã mount Drive hoặc đặt dữ liệu vào /content/datasets"
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
