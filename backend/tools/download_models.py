#!/usr/bin/env python3
"""
tools/download_models.py
========================
Pre-download all ML model weights so the Pi does not need internet access
during real-time authentication.

Run this ONCE on a machine that has internet access (e.g. your laptop),
then copy the entire backend/ folder to the Raspberry Pi.

Downloads:
  1. FaceNet weights (~90 MB)   -> ~/.cache/torch/hub/checkpoints/
  2. MediaPipe FaceLandmarker   -> backend/app/ml/model_store/face_landmarker.task

Usage (from the backend/ directory):
    python tools/download_models.py
"""

import sys
import os
import logging
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MODEL_STORE = Path(__file__).resolve().parent.parent / "app" / "ml" / "model_store"
_LANDMARKER_PATH = _MODEL_STORE / "face_landmarker.task"
_TORCH_CACHE = _MODEL_STORE / "torch_cache"
_LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def download_facenet():
    logger.info("Downloading FaceNet (InceptionResnetV1-vggface2) ...")
    try:
        os.environ.setdefault("TORCH_HOME", str(_TORCH_CACHE))
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained="vggface2").eval()
        logger.info(
            "FaceNet download complete. Parameters: %d",
            sum(p.numel() for p in model.parameters()),
        )
    except ImportError:
        logger.error("facenet-pytorch is not installed. Run: pip install facenet-pytorch")
        sys.exit(1)
    except Exception as exc:
        logger.error("FaceNet download failed: %s", exc)
        sys.exit(1)


def download_face_landmarker():
    logger.info("Checking MediaPipe FaceLandmarker model ...")
    _MODEL_STORE.mkdir(parents=True, exist_ok=True)

    if _LANDMARKER_PATH.exists():
        logger.info("face_landmarker.task already present (%d bytes).", _LANDMARKER_PATH.stat().st_size)
        return

    logger.info("Downloading face_landmarker.task (~3 MB) ...")
    try:
        def _progress(block, block_size, total):
            if total > 0:
                pct = min(100, block * block_size * 100 // total)
                print(f"\r  {pct}%", end="", flush=True)

        urllib.request.urlretrieve(_LANDMARKER_URL, _LANDMARKER_PATH, reporthook=_progress)
        print()  # newline after progress
        logger.info("face_landmarker.task saved to %s (%d bytes).", _LANDMARKER_PATH, _LANDMARKER_PATH.stat().st_size)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if _LANDMARKER_PATH.exists():
            _LANDMARKER_PATH.unlink()
        sys.exit(1)


def verify_mediapipe():
    logger.info("Verifying MediaPipe Tasks API ...")
    try:
        import mediapipe as mp
        _ = mp.tasks.vision.FaceLandmarker
        logger.info("MediaPipe Tasks API: OK  (version %s)", mp.__version__)
    except (ImportError, AttributeError) as exc:
        logger.error("MediaPipe check failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    download_face_landmarker()
    verify_mediapipe()
    download_facenet()
    logger.info("All models are ready. You can now run the system offline.")
