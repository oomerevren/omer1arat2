#!/usr/bin/env bash
# Deep-Pipeline — Lambda Labs / Vast.ai Remote GPU Setup (Aşama 5.3)
# Kullanım: ssh root@REMOTE_IP 'bash -s' < scripts/setup_remote_gpu.sh

set -euo pipefail

REPO_URL="https://github.com/oomerevren/deeppipelinedsad.git"
REMOTE_DIR="~/deeppipelinedsad"

echo "=== Deep-Pipeline Remote GPU Setup ==="

# 1. Sistem güncelleme
sudo apt-get update -qq && sudo apt-get install -y -qq git curl rsync htop

# 2. Miniconda (opsiyonel) veya sistem Python
if ! command -v python3 &>/dev/null; then
    echo "Python3 bulunamadı. Yükleniyor..."
    sudo apt-get install -y -qq python3 python3-pip python3-venv
fi

# 3. Repo klonla
if [ -d "$REMOTE_DIR" ]; then
    echo "Repo zaten var, güncelleniyor..."
    cd "$REMOTE_DIR" && git pull
else
    git clone "$REPO_URL" "$REMOTE_DIR"
    cd "$REMOTE_DIR"
fi

# 4. Bağımlılıkları kur
echo "Bağımlılıklar kuruluyor..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 5. GPU kontrol
echo "=== GPU Kontrol ==="
nvidia-smi || echo "UYARI: nvidia-smi bulunamadı — GPU sürücüleri eksik olabilir."
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK')"

# 6. Veri dizinini hazırla
mkdir -p data experiments/outputs submissions

echo "=== Kurulum tamamlandı ==="
echo "Veriyi rsync ile yükle:"
echo "  rsync -avz --progress local-data/ root@REMOTE_IP:~/deeppipelinedsad/data/"
echo "Eğitimi başlat:"
echo "  ssh root@REMOTE_IP 'cd ~/deeppipelinedsad && python scripts/run_experiment.py --config configs/model/kaggle.yaml --mode kaggle'"
echo "Sonuçları al:"
echo "  rsync -avz root@REMOTE_IP:~/deeppipelinedsad/experiments/outputs/ local-outputs/"
