#!/usr/bin/env bash
# ==============================================================================
# SCRIPT 1 BƯỚC DUY NHẤT CHẠY INFER LGTRACK TRÊN KAGGLE CHO TẬP UAV-ANTI-UAV
# Sử dụng trên Kaggle Notebook:
#   !bash run_kaggle.sh [--max_seqs N]
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=================================================================="
echo "          KIỂM TRA HỆ THỐNG & CẤU HÌNH KAGGLE NOTEBOOK"
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
# Tự động tải weights nếu chưa có
if [ -f "$DIR/checkpoints/download_weights.sh" ]; then
    bash "$DIR/checkpoints/download_weights.sh"
fi

# Cấu hình Torch Hub cache để sử dụng ngay weights local, không tải lại từ Facebook Research
mkdir -p "$HOME/.cache/torch/hub/checkpoints"
cp -n "$DIR/checkpoints/"*.pth "$HOME/.cache/torch/hub/checkpoints/" 2>/dev/null || true

# 4. Tìm kiếm và chuẩn bị tập dữ liệu UAV-Anti-UAV
echo ""
echo "[3/4] Tìm kiếm tập dữ liệu UAV-Anti-UAV..."

TARGET_DATA_DIR="/kaggle/input/datasets/yolandcandy/uav-anti-uav/Test-002/Test"
DATA_DIR="$TARGET_DATA_DIR"

# 4.1 Ưu tiên 1: Kiểm tra trực tiếp đường dẫn chỉ định
if [ -d "$TARGET_DATA_DIR" ]; then
    echo "[+] Tìm thấy dataset tại: $DATA_DIR"
elif [ -d "/kaggle/input" ]; then
    # Kiểm tra trực tiếp các đường dẫn liên quan
    KNOWN_PATHS=(
        "$TARGET_DATA_DIR"
        "/kaggle/input/datasets/yolandcandy/uav-anti-uav/Test-002"
        "/kaggle/input/datasets/yolandcandy/uav-anti-uav"
        "/kaggle/input/datasets/huynhat15/uavantiuav-test"
        "/kaggle/input/uavantiuav-test"
        "/kaggle/input/uav-anti-uav"
        "/kaggle/input/anti-uav"
    )
    for kp in "${KNOWN_PATHS[@]}"; do
        if [ -d "$kp/Test" ] || [ -d "$kp/Train" ] || [ -d "$kp" ]; then
            DATA_DIR="$kp"
            echo "[+] Tìm thấy dataset tại: $DATA_DIR"
            break
        fi
    done

    # Nếu chưa thấy, tìm kiếm đệ quy mọi thư mục "Test" trong /kaggle/input (tối đa 5 tầng)
    if [ ! -d "$DATA_DIR" ]; then
        for test_d in $(find /kaggle/input -maxdepth 5 -type d -name "Test" 2>/dev/null); do
            p_dir=$(dirname "$test_d")
            # Kiểm tra xem thư mục Test có chứa chuỗi video không
            if [ -n "$(ls -A "$test_d" 2>/dev/null)" ]; then
                DATA_DIR="$test_d"
                echo "[+] Tìm thấy dataset đã giải nén sẵn tại: $DATA_DIR"
                break
            fi
        done
    fi

    # 4.2 Ưu tiên 2: Nếu chỉ có file .zip trong /kaggle/input, giải nén sang /tmp/datasets
    if [ ! -d "$DATA_DIR" ]; then
        ANTI_ZIP=""
        for f in $(find /kaggle/input -name "*Anti-UAV*.zip" -o -name "*anti_uav*.zip" -o -name "*Test*.zip" 2>/dev/null); do
            if [ -f "$f" ]; then
                ANTI_ZIP="$f"
                break
            fi
        done
        if [ -n "$ANTI_ZIP" ]; then
            echo "[+] Tìm thấy file zip tại: $ANTI_ZIP"
            echo "[+] Đang giải nén sang /tmp/datasets/UAV-Anti-UAV..."
            mkdir -p /tmp/datasets/UAV-Anti-UAV
            unzip -q -o "$ANTI_ZIP" -d /tmp/datasets/UAV-Anti-UAV/
            DATA_DIR="/tmp/datasets/UAV-Anti-UAV"
            echo "[+] Giải nén hoàn tất vào: $DATA_DIR"
        fi
    fi
fi

# 4.3 Ưu tiên 3: Tìm kiếm tại các đường dẫn thông thường khác nếu chưa thấy
if [ ! -d "$DATA_DIR" ]; then
    FALLBACK_DIRS=(
        "/kaggle/working/datasets"
        "/tmp/datasets"
        "$DIR/data"
        "/content/datasets"
        "/home/nvidia/datasets"
    )
    for d in "${FALLBACK_DIRS[@]}"; do
        if [ -d "$d/UAV-Anti-UAV" ] || [ -d "$d/Test" ]; then
            DATA_DIR="$d"
            echo "[+] Tìm thấy dataset tại: $DATA_DIR"
            break
        fi
    done
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "[!] Chú ý: Chưa tìm thấy thư mục cục bộ, sử dụng đường dẫn Kaggle chỉ định: $TARGET_DATA_DIR"
    DATA_DIR="$TARGET_DATA_DIR"
fi

# Thiết lập thư mục lưu kết quả phù hợp với môi trường Kaggle
OUTPUT_DIR="/kaggle/working/results"
if [ ! -d "/kaggle/working" ]; then
    OUTPUT_DIR="$DIR/results"
fi
mkdir -p "$OUTPUT_DIR"

# 5. Chạy benchmark đánh giá CHUYÊN BIỆT cho mô hình LGTrack trên UAV-Anti-UAV
echo ""
echo "[4/4] Bắt đầu chạy benchmark đánh giá tập UAV-Anti-UAV với mô hình LGTrack..."
python3 evaluate.py --dataset anti_uav --models lgtrack --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" "$@"

echo ""
echo "=================================================================="
echo "      HOÀN THÀNH ĐÁNH GIÁ MÔ HÌNH LGTRACK TRÊN UAV-ANTI-UAV!"
echo " Báo cáo kết quả được lưu tại: $OUTPUT_DIR"
echo "=================================================================="

# Nén tự động thư mục results để người dùng tải về 1-click trên Kaggle Output
if [ -d "$OUTPUT_DIR" ] && [ -d "/kaggle/working" ]; then
    cd /kaggle/working
    zip -q -r benchmark_lgtrack_anti_uav_results.zip results/ 2>/dev/null || true
    if [ -f "/kaggle/working/benchmark_lgtrack_anti_uav_results.zip" ]; then
        echo "[+] Đã đóng gói sẵn file tải về: /kaggle/working/benchmark_lgtrack_anti_uav_results.zip"
    fi
fi
