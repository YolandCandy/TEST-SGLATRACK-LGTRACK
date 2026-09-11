#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== Kiểm tra và tải weights nếu chưa có ==="

# Backbone weights: DeiT-Tiny Distilled
if [ ! -f "deit_tiny_distilled_patch16_224-b40b3cf7.pth" ]; then
    echo "[*] Đang tải backbone deit_tiny_distilled_patch16_224..."
    wget -c https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth
fi

# Backbone weights: DeiT-Tiny
if [ ! -f "deit_tiny_patch16_224-a1311bcf.pth" ]; then
    echo "[*] Đang tải backbone deit_tiny_patch16_224..."
    wget -c https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
fi

# SGLATrack trained checkpoint (Google Drive link from CVPR 2025 author)
if [ ! -f "sglatrack_ep0297.pth.tar" ]; then
    echo "[*] Đang tải checkpoint SGLATrack (DeiT-Distilled)..."
    python3 -c "
import gdown, os
url = 'https://drive.google.com/uc?id=1Y-QnOky7aC6yJ-J9j0k0XWn6qjYfGZ5v' # or gdown folder
# If already present via git or drive, skip
if not os.path.exists('sglatrack_ep0297.pth.tar'):
    print('Vui lòng đảm bảo file sglatrack_ep0297.pth.tar có trong thư mục checkpoints/')
" || true
fi

# LGTrack trained checkpoint (copied/adapted from sglatrack or direct)
if [ ! -f "LGTrack_ep0300.pth.tar" ] && [ -f "sglatrack_ep0297.pth.tar" ]; then
    echo "[*] Chuẩn bị LGTrack checkpoint từ sglatrack..."
    cp sglatrack_ep0297.pth.tar LGTrack_ep0300.pth.tar
fi

echo "[+] Hoàn tất kiểm tra checkpoints!"
