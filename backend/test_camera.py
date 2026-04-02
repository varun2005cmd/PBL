#!/usr/bin/env python3
# =============================================================================
# backend/test_camera.py
#
# Standalone face-recognition test for Raspberry Pi 5.
# NO dashboard, NO servo, NO LCD required.
#
# Flow per cycle:
#   1. Wait until a face appears in the camera frame
#   2. Issue a liveness challenge (head turn LEFT or RIGHT)
#   3. Capture post-turn frame, verify the head actually turned
#   4. Run FaceNet embedding + SVM recognition
#   5. Print GRANTED / DENIED to the terminal
#   6. Optionally save a JPEG snapshot
#
# Usage (run from the backend/ directory):
#   python test_camera.py
#
# Environment variables:
#   CAMERA_INDEX=0        USB camera device index (default 0)
#   NO_LIVENESS=1         Skip the head-turn challenge (quick test)
#   SAVE_SNAPSHOTS=1      Save a JPEG per detection to backend/snapshots/
# =============================================================================

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_camera")

CAMERA_INDEX   = int(os.environ.get("CAMERA_INDEX",   "0"))
NO_LIVENESS    = os.environ.get("NO_LIVENESS",    "0") == "1"
SAVE_SNAPSHOTS = os.environ.get("SAVE_SNAPSHOTS", "0") == "1"
SNAPSHOT_DIR   = BACKEND_DIR / "snapshots"

if SAVE_SNAPSHOTS:
    SNAPSHOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Open camera
# ---------------------------------------------------------------------------
def open_camera():
    import cv2
    logger.info("Opening camera (index %d) ...", CAMERA_INDEX)

    # Use V4L2 on Linux (Pi); fall back to default backend on Windows
    import platform
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)   # fallback: no V4L2 flag

    if not cap.isOpened():
        logger.error(
            "Cannot open camera at index %d.\n"
            "  Check: lsusb                 (Logitech should appear)\n"
            "  Check: ls /dev/video*         (expect /dev/video0)\n"
            "  Try:   CAMERA_INDEX=1 python test_camera.py",
            CAMERA_INDEX,
        )
        sys.exit(1)

    # Warm up the camera (first few frames are often blank)
    for _ in range(5):
        cap.grab()

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Camera opened but cannot read frames. Check USB connection.")
        sys.exit(1)

    h, w = frame.shape[:2]
    logger.info("Camera OK  --  resolution %dx%d", w, h)
    return cap


# ---------------------------------------------------------------------------
# 2. Load ML models
# ---------------------------------------------------------------------------
def load_models():
    logger.info("Warming up MediaPipe face detector ...")
    from app.ml.face_detector import warmup
    warmup()
    logger.info("Face detector ready.")

    logger.info("Loading FaceNet embedding model (first run may take 30-60 s) ...")
    from app.ml.embedder import _get_model
    _get_model()
    logger.info("FaceNet ready.")


# ---------------------------------------------------------------------------
# 3. Load enrolled users from the SQLite database
# ---------------------------------------------------------------------------
def load_users(app):
    with app.app_context():
        from app.routes.user_routes import _load_prototype_embeddings
        prototypes = _load_prototype_embeddings()

    if not prototypes:
        logger.warning(
            "No enrolled users found in the database.\n"
            "  Enroll users first:\n"
            "    python enroll_user.py --name 'Your Name'"
        )
    else:
        logger.info(
            "Loaded %d enrolled user(s): %s",
            len(prototypes), list(prototypes.keys()),
        )
    return prototypes


# ---------------------------------------------------------------------------
# 4. Recognition loop
# ---------------------------------------------------------------------------
def run_test_loop(cap, app):
    import cv2
    from app.ml.face_detector import detect_face
    from app.ml.embedder      import generate_embedding
    from app.ml.recognizer    import recognize_user
    from app.ml.liveness      import generate_challenge, check_liveness
    from app.models           import db
    from app.models.log       import AccessLog

    logger.info("=" * 58)
    logger.info("Face recognition test running. Stand in front of the camera.")
    if NO_LIVENESS:
        logger.info("Liveness check DISABLED (NO_LIVENESS=1)")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 58)

    attempt = 0

    while True:
        # Reload prototypes on each attempt so new enrolments take effect
        # without restarting the script.
        with app.app_context():
            from app.routes.user_routes import _load_prototype_embeddings
            prototypes = _load_prototype_embeddings()

        # -- Wait for a face -------------------------------------------------
        cap.grab()   # flush stale buffered frame
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.15)
            continue

        detection = detect_face(frame)
        if detection is None:
            time.sleep(0.15)
            continue

        attempt += 1
        logger.info("--- Attempt #%d ---", attempt)
        logger.info("Face detected  bbox=%s", detection["bbox"])

        # -- Liveness --------------------------------------------------------
        if NO_LIVENESS:
            liveness_passed = True
        else:
            token     = generate_challenge()
            challenge = token["challenge"]
            issued_at = token["issued_at"]

            print(f"\n  >>> LIVENESS CHALLENGE: Turn your head {challenge} <<<\n")
            time.sleep(3.0)

            cap.grab()
            ret2, frame2 = cap.read()
            if not ret2 or frame2 is None:
                logger.warning("Could not read liveness response frame. Retrying.")
                time.sleep(1)
                continue

            det2 = detect_face(frame2)
            if det2 is None:
                logger.warning("No face in liveness response frame. Retrying.")
                time.sleep(1)
                continue

            liveness_result = check_liveness(
                frame=frame2,
                challenge=challenge,
                landmarks=det2["landmarks"],
                challenge_issued_at=issued_at,
            )
            liveness_passed = liveness_result["passed"]
            logger.info(
                "Liveness: passed=%s  yaw=%.1f deg  reason=%s",
                liveness_passed, liveness_result["yaw"], liveness_result["reason"],
            )
            if liveness_passed:
                detection = det2
                frame     = frame2

        if not liveness_passed:
            logger.info("RESULT: DENIED  --  liveness failed\n")
            _save_snapshot(frame, "DENIED_liveness", attempt)
            time.sleep(2)
            continue

        # -- Embedding -------------------------------------------------------
        embedding = generate_embedding(detection["face_crop"])
        if embedding is None:
            logger.warning("Could not generate embedding. Skipping.")
            time.sleep(1)
            continue

        # -- Recognition -----------------------------------------------------
        if not prototypes:
            logger.info("RESULT: No users enrolled, cannot recognise.\n")
            time.sleep(2)
            continue

        rec        = recognize_user(embedding, prototypes)
        user       = rec["user"]
        confidence = rec["confidence"]
        distance   = rec["distance"]
        method     = rec["method"]

        if user == "unknown":
            logger.info(
                "RESULT: DENIED   unknown face  (dist=%.3f, threshold=0.9)\n",
                distance,
            )
            _save_snapshot(frame, "DENIED_unknown", attempt)
        else:
            logger.info(
                "RESULT: GRANTED  --  %s  (confidence=%.1f%%  dist=%.3f  method=%s)\n",
                user, confidence * 100, distance, method,
            )
            _save_snapshot(frame, f"GRANTED_{user}", attempt)

        # -- Log to DB -------------------------------------------------------
        with app.app_context():
            log = AccessLog(
                timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                status     = "granted" if user != "unknown" else "denied",
                user       = user,
                liveness   = liveness_passed,
                confidence = confidence,
                detail     = f"test_camera  dist={distance:.3f}  {method}",
                ip_address = "localhost",
            )
            db.session.add(log)
            db.session.commit()

        time.sleep(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_snapshot(frame, label: str, attempt: int):
    if not SAVE_SNAPSHOTS:
        return
    import cv2
    ts   = datetime.now().strftime("%H%M%S")
    path = SNAPSHOT_DIR / f"{attempt:04d}_{ts}_{label}.jpg"
    cv2.imwrite(str(path), frame)
    logger.info("Snapshot: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 58)
    print("  Door Lock -- Camera Test Mode")
    print("  No dashboard / no hardware wiring required")
    print("=" * 58 + "\n")

    cap = open_camera()
    load_models()

    from app.app import create_app
    flask_app = create_app()

    try:
        run_test_loop(cap, flask_app)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        cap.release()
        logger.info("Camera released.")


if __name__ == "__main__":
    main()
