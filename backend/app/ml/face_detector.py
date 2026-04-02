# =============================================================================
# app/ml/face_detector.py
# Face Detection using MediaPipe FaceLandmarker (Tasks API)
#
# Uses the MediaPipe Tasks API (mediapipe >= 0.10) which works on Python 3.14+
# and on Raspberry Pi 5 aarch64.
#
# Requires the face_landmarker.task model file (downloaded by
# tools/download_models.py).
#
# Install: pip install mediapipe
#
# Provides detect_face(frame) which returns:
#   {
#     "face_crop": np.ndarray (aligned face, 160x160 RGB),
#     "bbox":      [x1, y1, x2, y2],
#     "landmarks": {"left_eye": (x,y), "right_eye": (x,y),
#                   "nose": (x,y), "mouth_left": (x,y), "mouth_right": (x,y)},
#     "confidence": float,
#   }
# Returns None if no face is detected.
# =============================================================================

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Model file location  (downloaded once by tools/download_models.py)
# --------------------------------------------------------------------------
_MODEL_PATH = Path(__file__).parent / "model_store" / "face_landmarker.task"

# MediaPipe FaceLandmarker landmark indices for 5 keypoints used by liveness.py
# (same indices as before — unchanged)
_MP_LEFT_EYE    = 33    # outer left eye corner (subject's left)
_MP_RIGHT_EYE   = 263   # outer right eye corner (subject's right)
_MP_NOSE        = 1     # nose tip
_MP_MOUTH_LEFT  = 61    # left mouth corner
_MP_MOUTH_RIGHT = 291   # right mouth corner

_landmarker = None   # cached FaceLandmarker instance


def _get_detector():
    """Lazy-load and cache the MediaPipe FaceLandmarker (Tasks API)."""
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    try:
        import mediapipe as mp
        BaseOptions          = mp.tasks.BaseOptions
        FaceLandmarker       = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode    = mp.tasks.vision.RunningMode
    except ImportError as exc:
        raise ImportError(
            "mediapipe is not installed. Run: pip install mediapipe"
        ) from exc

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {_MODEL_PATH}.\n"
            "Run:  python tools/download_models.py"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def warmup() -> None:
    """
    Force-load the MediaPipe model before the door loop starts.
    Call once at application startup to avoid a slow first authentication.
    """
    try:
        _get_detector()
        logger.info("MediaPipe FaceLandmarker warmed up.")
    except Exception as exc:
        logger.warning("Face detector warmup failed: %s", exc)


def release() -> None:
    """Release MediaPipe resources. Call at application shutdown."""
    global _landmarker
    if _landmarker is not None:
        try:
            _landmarker.close()
        except Exception:
            pass
        _landmarker = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_face(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Detect a single face in *frame* using MediaPipe FaceLandmarker.

    Parameters
    ----------
    frame : np.ndarray
        BGR image as returned by OpenCV (cv2.VideoCapture.read()).

    Returns
    -------
    dict | None
        Detection result with keys:
          - face_crop  : RGB np.ndarray, shape (160, 160, 3), uint8
          - bbox       : [x1, y1, x2, y2]  (pixel coords)
          - landmarks  : dict with 5 named keypoints (pixel coords)
          - confidence : float [0, 1]
        Returns None when no face detected.
    """
    import mediapipe as mp

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    landmarker = _get_detector()
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result     = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face_lm = result.face_landmarks[0]   # list of NormalizedLandmark

    def _px(lm_idx: int):
        lm = face_lm[lm_idx]
        return (int(lm.x * w), int(lm.y * h))

    landmarks = {
        "left_eye":    _px(_MP_LEFT_EYE),
        "right_eye":   _px(_MP_RIGHT_EYE),
        "nose":        _px(_MP_NOSE),
        "mouth_left":  _px(_MP_MOUTH_LEFT),
        "mouth_right": _px(_MP_MOUTH_RIGHT),
    }

    # Derive bounding box from all landmark x/y coords
    xs = [int(lm.x * w) for lm in face_lm]
    ys = [int(lm.y * h) for lm in face_lm]
    pad = int(0.15 * (max(xs) - min(xs)))   # add 15% padding around face
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)

    face_region = rgb[y1:y2, x1:x2]
    if face_region.size == 0:
        return None
    face_crop = cv2.resize(face_region, (160, 160))

    return {
        "face_crop":  face_crop,          # RGB (160, 160, 3)
        "bbox":       [x1, y1, x2, y2],
        "landmarks":  landmarks,
        "confidence": 1.0,               # MediaPipe Tasks API does not expose a scalar score
    }
