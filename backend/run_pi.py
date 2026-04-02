#!/usr/bin/env python3
# =============================================================================
# run_pi.py
# Raspberry Pi 5 – Combined launcher
#
# Starts two things in one process:
#   1. Flask HTTP server (API + dashboard backend) on port 5000
#   2. Physical door loop (camera -> ML pipeline -> LCD + servo)
#
# Usage:
#   cd backend
#   python run_pi.py
#
# Environment variables (all optional):
#   CAMERA_INDEX   USB camera device index (default: 0)
#   CAMERA_WIDTH   Capture width  in pixels (default: 1280)
#   CAMERA_HEIGHT  Capture height in pixels (default: 720)
#   CAMERA_FPS     Capture frame rate       (default: 30)
#   FLASK_HOST     IP to bind Flask to      (default: 0.0.0.0)
#   FLASK_PORT     Port for Flask           (default: 5000)
# =============================================================================

import logging
import os
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from app.app import create_app
    from app.models import db

    app = create_app()

    # Ensure DB tables exist once before threads start
    with app.app_context():
        db.create_all()

    # ------------------------------------------------------------------
    # Warm up ML models BEFORE starting the door loop thread so the first
    # real authentication attempt is not penalised by model-load latency.
    # FaceNet (~100 MB) and MediaPipe are loaded into memory here once.
    # ------------------------------------------------------------------
    logger.info("Warming up ML models (this may take 20-40 s on first run) …")
    try:
        from app.ml.face_detector import warmup as detector_warmup
        from app.ml.embedder import _get_model as _load_facenet
        detector_warmup()
        _load_facenet()        # downloads weights if not cached, then holds in RAM
        logger.info("ML models ready.")
    except Exception as exc:
        logger.warning("Model warmup warning (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Thread 1: Physical door loop
    # ------------------------------------------------------------------
    from app.hardware.door_loop import run_door_loop

    door_thread = threading.Thread(
        target=run_door_loop,
        args=(app,),
        daemon=True,
        name="door-loop",
    )
    door_thread.start()
    logger.info("Door loop thread started.")

    # ------------------------------------------------------------------
    # Thread 2: Flask HTTP server (blocking – runs in main thread)
    # ------------------------------------------------------------------
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    logger.info("Starting Flask server on %s:%d …", host, port)
    # use_reloader=False is required when running inside a thread
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
