#!/usr/bin/env python3
# =============================================================================
# run_pi.py
# Raspberry Pi 5 – Combined launcher
# =============================================================================

import os
import sys

# ---------------------------------------------------------------------------
# CRITICAL FIX: STRANGLE C++ THREAD CONTENTION
# Must be set BEFORE importing numpy, cv2, torch, or mediapipe.
# Prevents segmentation faults and CPU spiking on Raspberry Pi 5.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import logging
import threading
import cv2

# Force OpenCV to run single-threaded (prevents TBB/pthread contention)
cv2.setNumThreads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    from app.app import create_app
    from app.models import db

    app = create_app()

    # Ensure DB tables exist
    with app.app_context():
        db.create_all()

    # ------------------------------------------------------------------
    # Warm up ML models strictly on the main thread
    # ------------------------------------------------------------------
    logger.info("Warming up ML models (this may take 20-40 s) …")
    try:
        import torch
        # Force PyTorch CPU backend to single-thread mode
        torch.set_num_threads(1)
        
        from app.ml.face_detector import warmup as detector_warmup
        from app.ml.embedder import warmup as embedder_warmup
        detector_warmup()
        embedder_warmup()
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
    # Thread 2: Flask HTTP server
    # ------------------------------------------------------------------
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    logger.info("Starting Flask server on %s:%d …", host, port)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    main()
