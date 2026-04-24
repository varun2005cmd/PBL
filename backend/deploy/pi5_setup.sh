#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "[1/7] Installing system dependencies"
sudo apt-get update
sudo apt-get install -y \
  python3.11 python3.11-venv python3-pip \
  build-essential pkg-config git curl \
  libatlas-base-dev libopenblas-dev liblapack-dev gfortran \
  libjpeg-dev zlib1g-dev libtiff5-dev libopenjp2-7-dev libpng-dev \
  libavcodec-dev libavformat-dev libswscale-dev libglib2.0-0 libgl1 \
  v4l-utils i2c-tools \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

echo "[2/7] Enabling I2C interface"
sudo raspi-config nonint do_i2c 0 || true

echo "[3/7] Creating Python 3.11 virtual environment"
cd "$BACKEND_DIR"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "[4/7] Installing Python dependencies"
pip install -r ../requirements.txt

echo "[5/7] Downloading ML models"
python tools/download_models.py

echo "[6/7] Initializing database"
python - <<'PY'
from app.app import create_app
from app.models import db
app = create_app()
with app.app_context():
    db.create_all()
print('Database initialized')
PY

echo "[7/7] Setup complete"
echo "Run manually: cd $BACKEND_DIR && source .venv/bin/activate && python run_pi.py"
