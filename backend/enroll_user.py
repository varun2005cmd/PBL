#!/usr/bin/env python3
# =============================================================================
# backend/enroll_user.py
#
# CLI enrollment tool for Raspberry Pi 5.
# Captures face photos via USB camera, computes FaceNet embeddings,
# stores them in the SQLite DB, and retrains the SVM classifier.
# NO dashboard required.
#
# Usage (run from the backend/ directory):
#   python enroll_user.py --name "Palash"
#   python enroll_user.py --name "Palash" --num-photos 15 --delay 2.0
#
# Options:
#   --name         User name (required)
#   --num-photos   Number of face photos to capture (default: 10)
#   --delay        Seconds between captures (default: 1.5)
#   --camera       Camera device index (default: 0, or CAMERA_INDEX env var)
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enroll_user")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def open_camera(index: int):
    import cv2
    logger.info("Opening camera (index %d) ...", index)

    # Use V4L2 on Linux (Pi); fall back to default backend on Windows
    import platform
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        logger.error(
            "Cannot open camera at index %d. Check lsusb and ls /dev/video*", index
        )
        sys.exit(1)

    for _ in range(5):
        cap.grab()

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Camera opened but cannot read frames.")
        sys.exit(1)

    h, w = frame.shape[:2]
    logger.info("Camera OK  --  %dx%d", w, h)
    return cap


# ---------------------------------------------------------------------------
# ML warmup
# ---------------------------------------------------------------------------
def warmup_models():
    logger.info("Warming up MediaPipe face detector ...")
    from app.ml.face_detector import warmup
    warmup()

    logger.info("Loading FaceNet (may take 30-60 s first run) ...")
    from app.ml.embedder import _get_model
    _get_model()

    logger.info("Models ready.")


# ---------------------------------------------------------------------------
# Main enrollment routine
# ---------------------------------------------------------------------------
def enroll(name: str, num_photos: int, delay: float, camera_index: int):
    from app.app import create_app
    from app.models import db
    from app.models.user import User
    from app.ml.face_detector import detect_face
    from app.ml.embedder import generate_embedding

    flask_app = create_app()

    # -- Get or create the user in the DB ------------------------------------
    with flask_app.app_context():
        user = User.query.filter_by(name=name).first()
        if user is None:
            user = User(name=name)
            db.session.add(user)
            db.session.commit()
            logger.info("Created new user '%s' (id=%d)", user.name, user.id)
        else:
            logger.info("Found existing user '%s' (id=%d)", user.name, user.id)
        user_id = user.id

    cap = open_camera(camera_index)

    print("\n" + "=" * 56)
    print(f"  Enrolling: {name}")
    print(f"  Photos   : {num_photos}  (delay {delay}s between each)")
    print(f"  Tip      : keep your face centred, vary angle slightly")
    print("=" * 56 + "\n")
    input("  Press ENTER when ready, then stay still ...")
    print()

    captured     = 0
    attempt      = 0
    embeddings_collected: list = []

    while captured < num_photos:
        attempt += 1
        time.sleep(delay)

        cap.grab()
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("[%d] Frame capture failed. Retrying.", attempt)
            continue

        detection = detect_face(frame)
        if detection is None:
            logger.warning("[%d] No face detected. Move closer to the camera.", attempt)
            continue

        embedding = generate_embedding(detection["face_crop"])
        if embedding is None:
            logger.warning("[%d] Could not generate embedding. Retrying.", attempt)
            continue

        captured += 1
        embeddings_collected.append(embedding)
        logger.info("[%d] Photo %d/%d captured.  bbox=%s", attempt, captured, num_photos, detection["bbox"])

    cap.release()
    logger.info("Captured %d/%d photos successfully.", captured, num_photos)

    # -- Store embeddings in the DB ------------------------------------------
    with flask_app.app_context():
        user = db.session.get(User, user_id)

        # Load existing embeddings (if any) and append using the model's methods
        existing       = user.get_embeddings()   # returns list-of-lists
        new_embeddings = [e.tolist() for e in embeddings_collected]
        all_embeddings = existing + new_embeddings
        user.set_embeddings(all_embeddings)      # sets embeddings_json + enrolled flag
        db.session.commit()
        total = len(all_embeddings)
        logger.info("Stored %d embeddings for '%s' (total now: %d).", len(new_embeddings), name, total)

    # -- Retrain SVM ---------------------------------------------------------
    logger.info("Retraining SVM classifier ...")
    with flask_app.app_context():
        from app.routes.user_routes import _load_prototype_embeddings
        from app.ml.recognizer import train_classifier

        prototypes = _load_prototype_embeddings()
        result     = train_classifier(prototypes)

    if result["ok"]:
        logger.info("SVM retrained successfully. %s", result.get("message", ""))
    else:
        # With only 1 enrolled user, SVM can't train (needs >= 2 classes).
        # Euclidean nearest-neighbour recognition still works perfectly.
        msg = result.get("message", "") or result.get("detail", "")
        logger.info("SVM skipped (%s). Euclidean recognition is active.", msg)

    print("\n" + "=" * 56)
    print(f"  Enrollment complete for '{name}'.")
    print(f"  Run:  python test_camera.py  to verify recognition.")
    print("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enrol a user into the face-recognition door lock.")
    parser.add_argument("--name",       required=True,            help="Full name of the user to enrol")
    parser.add_argument("--num-photos", type=int,   default=10,   help="Number of face photos to capture (default: 10)")
    parser.add_argument("--delay",      type=float, default=1.5,  help="Seconds between captures (default: 1.5)")
    parser.add_argument("--camera",     type=int,
                        default=int(os.environ.get("CAMERA_INDEX", "0")),
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    warmup_models()
    enroll(
        name         = args.name,
        num_photos   = args.num_photos,
        delay        = args.delay,
        camera_index = args.camera,
    )


if __name__ == "__main__":
    main()
