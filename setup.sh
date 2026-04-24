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

PYTHON_BIN="python3.11"


echo "Using $PYTHON_BIN for virtual environment..."
$PYTHON_BIN -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "Installing pure-CPU Torch 2.2.2 to prevent NVIDIA downloads..."
python -m pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://www.piwheels.org/simple

echo "Installing MediaPipe..."
python -m pip install --extra-index-url https://www.piwheels.org/simple mediapipe || true

echo "Installing remaining project requirements..."
python -m pip install -r requirements.txt
# Install an older OpenCV that works with NumPy 1.x
python -m pip install --extra-index-url https://www.piwheels.org/simple "opencv-python-headless<4.11"

# Force NumPy 1.x (Everyone is happy now!)
python -m pip install "numpy<2"



python backend/tools/download_models.py || true

mkdir -p backend/logs backend/data/violations
if [ ! -f .env ]; then
