# =============================================================================
# app/ml/pipeline.py
# Main Authentication Pipeline
#
# authenticate_user() is the single entry-point used by the REST endpoint.
# It orchestrates:
#   1. Face detection          (MTCNN)
#   2. Liveness verification   (solvePnP yaw-based challenge-response)
#   3. Embedding generation    (FaceNet / InceptionResnetV1)
#   4. Identity recognition    (SVM + Euclidean distance)
#
# The function returns a JSON-serialisable dict:
#   {
#     "status":     "granted" | "denied",
#     "user":       "<name>" | "unknown",
#     "liveness":   true | false,
#     "confidence": <float 01>,
#     "detail":     "<human-readable explanation>"
#   }
#
# All heavy dependencies are imported lazily; the module can be imported
# at startup without blocking on model downloads.
# =============================================================================

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.ml.face_detector import detect_face
from app.ml.liveness     import check_liveness
from app.ml.embedder     import generate_embedding, list_to_embedding
from app.ml.recognizer   import recognize_user

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def authenticate_user(
    frame: np.ndarray,
    challenge: str,
    challenge_issued_at: float,
    prototype_embeddings: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """
    Full authentication pipeline for a single camera frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image as read by OpenCV from a camera or decoded JPEG/PNG.
    challenge : str
        The challenge direction issued earlier: ``"LEFT"`` or ``"RIGHT"``.
    challenge_issued_at : float
        ``time.time()`` value when the challenge was issued (for timeout check).
    prototype_embeddings : dict
        Enrolled user embeddings: ``{username: [emb, emb, ...]}``.
        Each ``emb`` is a (512,) float32 np.ndarray.
        Retrieved from the database by the route handler before calling this.

    Returns
    -------
    dict  (JSON-serialisable)
        {
          "status":     "granted" | "denied",
          "user":       str,
          "liveness":   bool,
          "confidence": float,
          "detail":     str
        }
    """
    # ------------------------------------------------------------------
    # Stage 1  Face Detection
    # ------------------------------------------------------------------
    detection = detect_face(frame)

    if detection is None:
        return _denied("no_face", "No face detected in the frame.", liveness=False)

    face_crop  = detection["face_crop"]    # (160, 160, 3) RGB
    landmarks  = detection["landmarks"]
    det_conf   = detection["confidence"]
    logger.debug("Face detected. MTCNN confidence=%.3f", det_conf)

    # ------------------------------------------------------------------
    # Stage 2  Liveness Check
    # ------------------------------------------------------------------
    liveness_result = check_liveness(
        frame=frame,
        challenge=challenge,
        landmarks=landmarks,
        challenge_issued_at=challenge_issued_at,
    )

    if not liveness_result["passed"]:
        return _denied(
            "liveness_fail",
            liveness_result["reason"],
            liveness=False,
        )

    logger.debug("Liveness passed. Yaw=%.1f", liveness_result["yaw"])

    # ------------------------------------------------------------------
    # Stage 3  Embedding Generation
    # ------------------------------------------------------------------
    embedding = generate_embedding(face_crop)

    if embedding is None:
        return _denied("embedding_fail", "Could not generate face embedding.", liveness=True)

    # ------------------------------------------------------------------
    # Stage 4  Identity Recognition
    # ------------------------------------------------------------------
    recognition = recognize_user(embedding, prototype_embeddings)
    user       = recognition["user"]
    confidence = recognition["confidence"]
    distance   = recognition["distance"]

    if user == "unknown":
        return {
            "status":     "denied",
            "user":       "unknown",
            "liveness":   True,
            "confidence": 0.0,
            "detail":     (
                f"Face not recognised (distance={distance:.3f}, "
                f"threshold={0.9})."
            ),
        }

    logger.info("Access granted: user=%s confidence=%.3f", user, confidence)
    return {
        "status":     "granted",
        "user":       user,
        "liveness":   True,
        "confidence": round(confidence, 4),
        "detail":     f"Access granted. Distance={distance:.3f}.",
    }


def decode_frame(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode a raw JPEG/PNG byte payload into an OpenCV BGR ndarray.

    Use this in the route handler to convert ``request.data`` or a
    multipart file upload into a frame suitable for ``authenticate_user()``.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes (JPEG / PNG / BMP).

    Returns
    -------
    np.ndarray | None
        BGR frame, or None if decoding fails.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size > 0 else None


def enroll_user(
    frames: List[np.ndarray],
    username: str,
) -> Dict[str, Any]:
    """
    Compute embeddings for a list of enrollment frames for a new user.

    Typically called with 310 frames captured during on-boarding.
    The caller is responsible for persisting the returned embeddings to DB.

    Parameters
    ----------
    frames : list of np.ndarray
        BGR frames captured during enrollment.
    username : str
        Identity label for this user.

    Returns
    -------
    dict
        {
          "ok": bool,
          "username": str,
          "embeddings": [[float, ...], ...],   # list of 512-D lists
          "count": int,
          "message": str
        }
    """
    from app.ml.embedder import embeddings_to_list

    embeddings_list = []
    failed = 0

    for frame in frames:
        det = detect_face(frame)
        if det is None:
            failed += 1
            continue
        emb = generate_embedding(det["face_crop"])
        if emb is not None:
            embeddings_list.append(embeddings_to_list(emb))
        else:
            failed += 1

    if not embeddings_list:
        return {
            "ok":         False,
            "username":   username,
            "embeddings": [],
            "count":      0,
            "message":    f"No valid faces found in any of {len(frames)} frames.",
        }

    return {
        "ok":         True,
        "username":   username,
        "embeddings": embeddings_list,
        "count":      len(embeddings_list),
        "message":    (
            f"Enrolled {len(embeddings_list)} embeddings "
            f"({failed} frames skipped)."
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _denied(
    reason_code: str,
    detail: str,
    liveness: bool = False,
) -> Dict[str, Any]:
    """Construct a standardised 'denied' response."""
    return {
        "status":     "denied",
        "user":       "unknown",
        "liveness":   liveness,
        "confidence": 0.0,
        "detail":     detail,
    }
