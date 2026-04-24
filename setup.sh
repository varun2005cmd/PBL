#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Updating system and installing base libraries for Raspberry Pi 5..."
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-opencv \
  libhdf5-dev libjpeg-dev libopenblas-dev \
  i2c-tools git

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi

echo "Using $PYTHON_BIN for virtual environment..."
$PYTHON_BIN -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "Installing pure-CPU Torch 2.2.2 to prevent NVIDIA downloads..."
python -m pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://www.piwheels.org/simple

echo "Installing MediaPipe..."
python -m pip install --extra-index-url https://www.piwheels.org/simple mediapipe || true

# Installing remaining project requirements
python -m pip install -r requirements.txt
python -m pip install --extra-index-url https://www.piwheels.org/simple opencv-python-headless

python backend/tools/download_models.py || true

mkdir -p backend/logs backend/data/violations
if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Setup complete. Edit .env if needed, then run ./run.sh"
