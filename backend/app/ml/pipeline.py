# =============================================================================
# app/ml/pipeline.py  –  v3 (Bulletproof)
#
# Fixes applied:
#   • Removed ThreadPoolExecutor (causes C++ extension segfaults on Pi 5)
#   • Execution is now strictly sequential on the calling thread.
#   • Relies on internal timeouts in camera.py and liveness.py to prevent hangs.
#   • Completely eliminates thread contention between OpenCV, Mediapipe, and Torch.
# =============================================================================

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.ml.face_detector import detect_face
from app.ml.liveness      import check_liveness
from app.ml.embedder      import generate_embedding
from app.ml.recognizer    import recognize_user

logger = logging.getLogger(__name__)


def _l2_normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    try:
        if vec is None:
            return None
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 1:
            return None
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n < 1e-8:
            return None
        return vec / n
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def authenticate_user(
    frame: np.ndarray,
    challenge: str,
    challenge_issued_at: float,
    prototype_embeddings: Dict[str, List[np.ndarray]],
    liveness_frames: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Full authentication pipeline for a single camera frame.
    Strictly sequential. No nested threading.
    """
    start = time.monotonic()

    # ------------------------------------------------------------------
    # Stage 1  Face Detection
    # ------------------------------------------------------------------
    try:
        detection = detect_face(frame)
    except Exception as exc:
        logger.error("Stage 1 detection raised: %s", exc)
        return _denied("detection_error", "Face detection failed internally.", liveness=False)

    if detection is None:
        return _denied("no_face", "No face detected in the frame.", liveness=False)

    if isinstance(detection, dict) and detection.get("error") == "multiple_faces":
        return _denied("multiple_faces", "Multiple faces detected. Please ensure only one face is in the frame.", liveness=False)

    if not isinstance(detection, dict):
        return _denied("invalid_detection", "Invalid face detection output.", liveness=False)

    face_crop = detection.get("face_crop")
    landmarks = detection.get("landmarks")

    if face_crop is None or not isinstance(landmarks, dict):
        return _denied("invalid_detection", "Incomplete face detection output.", liveness=False)

    logger.debug("Stage 1 OK (%.2f s)", time.monotonic() - start)

    # ------------------------------------------------------------------
    # Stage 2  Liveness Check
    # ------------------------------------------------------------------
    if os.environ.get("NO_LIVENESS", "0") == "1":
        liveness_result = {"passed": True, "yaw": 0.0, "reason": "Liveness skipped (NO_LIVENESS=1)"}
    else:
        try:
            liveness_result = check_liveness(
                frame=frame,
                challenge=challenge,
                landmarks=landmarks,
                challenge_issued_at=challenge_issued_at,
                frames=liveness_frames,
            )
        except Exception as exc:
            logger.error("Stage 2 liveness raised: %s", exc)
            return _denied("liveness_error", "Liveness check failed internally.", liveness=False)

    if not isinstance(liveness_result, dict) or not liveness_result.get("passed"):
        reason = liveness_result.get("reason", "Liveness check failed.") if isinstance(liveness_result, dict) else "Liveness check failed."
        return _denied("liveness_fail", reason, liveness=False)

    logger.debug("Stage 2 OK (%.2f s)", time.monotonic() - start)

    # ------------------------------------------------------------------
    # Stage 3  Embedding Generation (Multi-frame averaged)
    # ------------------------------------------------------------------
    try:
        frames_to_process = []
        if liveness_frames and len(liveness_frames) > 0:
            frames_to_process = liveness_frames[-2:]
        if frame is not None:
            frames_to_process.append(frame)

        valid_embeddings: List[np.ndarray] = []
        for f in frames_to_process:
            if f is None: continue
            det = detect_face(f)
            if isinstance(det, dict) and det.get("face_crop") is not None:
                emb = generate_embedding(det["face_crop"])
                emb = _l2_normalize(emb) if emb is not None else None
                if emb is not None:
                    valid_embeddings.append(emb)

        if not valid_embeddings:
            return _denied("embedding_fail", "Could not generate face embedding.", liveness=True)

        # Robust temporal aggregation: drop outlier frames before averaging.
        # This prevents a single blurry/partial frame from dominating the mean.
        embs = np.stack(valid_embeddings, axis=0).astype(np.float32)  # (n, 512)
        centroid = np.mean(embs, axis=0)
        centroid = _l2_normalize(centroid)
        if centroid is None:
            return _denied("embedding_fail", "Could not stabilize face embedding.", liveness=True)

        dists = np.linalg.norm(embs - centroid.reshape(1, -1), axis=1)

        outlier_q = float(os.environ.get("EMB_TEMPO_OUTLIER_Q", "0.80"))
        max_dist = float(os.environ.get("EMB_TEMPO_MAX_DIST", "0.95"))
        try:
            q_thresh = float(np.quantile(dists, outlier_q)) if len(dists) >= 3 else float(np.max(dists))
        except Exception:
            q_thresh = float(np.max(dists))

        keep_mask = (dists <= min(q_thresh, max_dist))
        keep = embs[keep_mask]
        if keep.shape[0] == 0:
            keep = embs

        avg_emb = np.mean(keep, axis=0)
        embedding = _l2_normalize(avg_emb)
        if embedding is None:
            return _denied("embedding_fail", "Could not finalize face embedding.", liveness=True)

    except Exception as exc:
        logger.error("Stage 3 embedding raised: %s", exc)
        return _denied("embedding_error", "Could not generate face embedding.", liveness=True)

    logger.debug("Stage 3 OK (%.2f s)", time.monotonic() - start)

    # ------------------------------------------------------------------
    # Stage 4  Identity Recognition
    # ------------------------------------------------------------------
    try:
        recognition = recognize_user(embedding, prototype_embeddings)
    except Exception as exc:
        logger.error("Stage 4 recognition raised: %s", exc)
        return _denied("recognition_error", "Face recognition failed (internal error).", liveness=True)

    user       = str(recognition.get("user")       or "unknown")
    confidence = float(recognition.get("confidence") or 0.0)
    distance   = float(recognition.get("distance")   or 9.9)

    logger.debug(
        "Stage 4 OK: user=%s confidence=%.3f distance=%.3f (%.2f s)",
        user, confidence, distance, time.monotonic() - start,
    )

    if user == "unknown":
        return {
            "status":     "denied",
            "user":       "unknown",
            "liveness":   True,
            "confidence": 0.0,
            "detail":     f"Face not recognised (distance={distance:.3f}, threshold={os.environ.get('DISTANCE_THRESHOLD', '0.85')}).",
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
    """Decode a raw JPEG/PNG byte payload into an OpenCV BGR ndarray."""
    if not image_bytes:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None
        return frame
    except Exception as exc:
        logger.error("decode_frame: exception: %s", exc)
        return None


def enroll_user(
    frames: List[np.ndarray],
    username: str,
) -> Dict[str, Any]:
    """
    Compute embeddings for a list of enrollment frames.
    Strictly sequential processing.
    """
    if not frames:
        return {
            "ok": False, "username": username,
            "embeddings": [], "count": 0,
            "message": "No frames provided for enrollment.",
        }

    from app.ml.embedder import embeddings_to_list

    raw_embeddings: List[np.ndarray] = []
    failed = 0

    for frame in frames:
        if frame is None:
            failed += 1
            continue

        try:
            det = detect_face(frame)
        except Exception:
            failed += 1
            continue

        if not isinstance(det, dict) or det.get("error"):
            failed += 1
            continue

        crop = det.get("face_crop")
        try:
            emb = generate_embedding(crop)
        except Exception:
            failed += 1
            continue

        emb = _l2_normalize(emb) if emb is not None else None
        if emb is not None:
            raw_embeddings.append(emb)
        else:
            failed += 1

    if not raw_embeddings:
        return {
            "ok": False, "username": username,
            "embeddings": [], "count": 0,
            "message": f"No valid faces found in any of {len(frames)} frames.",
        }

    # Accuracy-first enrollment cleanup:
    # - compute a centroid template
    # - drop outlier embeddings (bad frames)
    # - keep only the most consistent samples to reduce noise
    embs = np.stack(raw_embeddings, axis=0).astype(np.float32)
    centroid = _l2_normalize(np.mean(embs, axis=0))
    if centroid is None:
        return {
            "ok": False, "username": username,
            "embeddings": [], "count": 0,
            "message": "Enrollment failed: could not stabilize embeddings.",
        }

    dists = np.linalg.norm(embs - centroid.reshape(1, -1), axis=1)
    outlier_q = float(os.environ.get("ENROLL_OUTLIER_Q", "0.85"))
    max_dist = float(os.environ.get("ENROLL_MAX_DIST", "0.90"))
    max_keep = int(os.environ.get("ENROLL_MAX_KEEP", "8"))

    try:
        q_thresh = float(np.quantile(dists, outlier_q)) if len(dists) >= 4 else float(np.max(dists))
    except Exception:
        q_thresh = float(np.max(dists))

    # Keep closest embeddings, but guarantee we keep at least 3 if possible.
    order = np.argsort(dists)
    keep_idx = [int(i) for i in order if float(dists[int(i)]) <= min(q_thresh, max_dist)]
    if len(keep_idx) < min(3, len(order)):
        keep_idx = [int(i) for i in order[: min(3, len(order))]]
    keep_idx = keep_idx[: max_keep]

    final_embeddings: List[np.ndarray] = [centroid]
    for i in keep_idx:
        final_embeddings.append(embs[i])

    embeddings_list = [embeddings_to_list(e) for e in final_embeddings]

    logger.info(
        "Enrollment complete: user=%s embeddings=%d failed=%d",
        username, len(embeddings_list), failed,
    )
    return {
        "ok":         True,
        "username":   username,
        "embeddings": embeddings_list,
        "count":      len(embeddings_list),
        "message":    f"Enrolled {len(embeddings_list)} embeddings ({failed} frames skipped).",
    }


def _denied(
    reason_code: str,
    detail: str,
    liveness: bool = False,
) -> Dict[str, Any]:
    """Construct a standardised 'denied' response dict."""
    logger.info("Auth denied: code=%s detail=%s", reason_code, detail)
    return {
        "status":     "denied",
        "user":       "unknown",
        "liveness":   liveness,
        "confidence": 0.0,
        "detail":     detail,
    }
