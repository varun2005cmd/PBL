#!/usr/bin/env bash
set -euo pipefail

# Force OpenCV to use stable backends on Pi 5
export OPENCV_VIDEOIO_PRIORITY_LIST=V4L2,GST,ANY
export TF_CPP_MIN_LOG_LEVEL=3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

mkdir -p backend/logs backend/data/violations
source venv/bin/activate
cd backend
exec python run_pi.py
