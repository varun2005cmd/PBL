#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-opencv \
  libhdf5-dev libatlas-base-dev libjpeg-dev libopenblas-dev \
  i2c-tools git

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --extra-index-url https://www.piwheels.org/simple mediapipe || true
python -m pip install -r requirements.txt

python backend/tools/download_models.py || true

mkdir -p backend/logs backend/data/violations
if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Setup complete. Edit .env if needed, then run ./run.sh"
