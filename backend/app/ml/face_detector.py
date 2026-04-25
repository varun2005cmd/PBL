# =============================================================================
# app/ml/face_detector.py  –  v2 (hardened)
#
# Fixes applied:
#   • _landmarker_lock protects BOTH lazy-init and detect() calls
#     (prevents race during parallel HTTP requests)
#   • _get_detector() is also guarded by the same lock (double-check pattern)
#   • All exceptions from mediapipe are caught and logged
#   • detect_face() validates frame BEFORE touching mediapipe
#   • face_region empty-array guard prevents crash in cv2.resize
#   • EAR calculation protected against zero-division and bad indices
#   • warmup() logs mediapipe version for diagnostics
# =============================================================================

import logging
import os
import threading
import warnings
from pathlib import Path

# Suppress Protobuf/MediaPipe deprecation warnings to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Quality gates (accuracy-first; conservative defaults; configurable via env)
# --------------------------------------------------------------------------

_MIN_FACE_BLUR_VAR = float(os.environ.get("MIN_FACE_BLUR_VAR", "22"))
_MIN_FACE_MEAN_L   = float(os.environ.get("MIN_FACE_MEAN_L", "32"))
_MAX_FACE_MEAN_L   = float(os.environ.get("MAX_FACE_MEAN_L", "225"))
_MIN_FACE_STD_L    = float(os.environ.get("MIN_FACE_STD_L", "16"))
_MIN_INTER_EYE_PX  = float(os.environ.get("MIN_INTER_EYE_PX", "16"))

_ALIGN_LEFT_EYE_X  = float(os.environ.get("ALIGN_LEFT_EYE_X", "0.35"))
_ALIGN_RIGHT_EYE_X = float(os.environ.get("ALIGN_RIGHT_EYE_X", "0.65"))
_ALIGN_EYE_Y       = float(os.environ.get("ALIGN_EYE_Y", "0.40"))


def _l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def _align_face_by_eyes(
    face_region: np.ndarray,
    rel_left_eye: tuple[int, int],
    rel_right_eye: tuple[int, int],
    out_size: int = 160,
) -> np.ndarray:
    """Return an aligned (out_size x out_size) RGB crop using the eye line."""
    lx, ly = rel_left_eye
    rx, ry = rel_right_eye

    # Validate eye coords
    h, w = face_region.shape[:2]
    if not (0 <= lx < w and 0 <= rx < w and 0 <= ly < h and 0 <= ry < h):
        raise ValueError("eye coords out of bounds")

    src_left = np.array([lx, ly], dtype=np.float32)
    src_right = np.array([rx, ry], dtype=np.float32)
    eye_vec = src_right - src_left
    src_dist = _l2(eye_vec)
    if src_dist < _MIN_INTER_EYE_PX:
        raise ValueError("inter-eye distance too small")

    desired_left = np.array([
        _ALIGN_LEFT_EYE_X * out_size,
        _ALIGN_EYE_Y * out_size,
    ], dtype=np.float32)
    desired_right = np.array([
        _ALIGN_RIGHT_EYE_X * out_size,
        _ALIGN_EYE_Y * out_size,
    ], dtype=np.float32)
    desired_dist = _l2(desired_right - desired_left)

    # Angle between eyes in the source image
    angle = float(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))
    scale = float(desired_dist / max(src_dist, 1e-6))

    # Rotate/scale around the left eye, then translate so it lands on desired_left.
    M = cv2.getRotationMatrix2D((float(src_left[0]), float(src_left[1])), -angle, scale)
    M[0, 2] += float(desired_left[0] - src_left[0])
    M[1, 2] += float(desired_left[1] - src_left[1])

    aligned = cv2.warpAffine(
        face_region,
        M,
        (out_size, out_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned


def _passes_quality_gates(face_crop: np.ndarray) -> bool:
    """Reject crops that are extremely blurred/dark/washed-out (accuracy-first)."""
    try:
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            return False
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        mean_l = float(gray.mean())
        std_l = float(gray.std())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur < _MIN_FACE_BLUR_VAR:
            return False
        if std_l < _MIN_FACE_STD_L:
            return False
        if mean_l < _MIN_FACE_MEAN_L or mean_l > _MAX_FACE_MEAN_L:
            return False
        return True
    except Exception:
        return True  # fail-open on unexpected cv2/platform issues

# --------------------------------------------------------------------------
# Model file location  (downloaded once by tools/download_models.py)
# --------------------------------------------------------------------------
_MODEL_PATH = Path(__file__).parent / "model_store" / "face_landmarker.task"

# MediaPipe FaceLandmarker landmark indices
_MP_LEFT_EYE    = 33    # outer left eye corner (subject's left)
_MP_RIGHT_EYE   = 263   # outer right eye corner (subject's right)
_MP_NOSE        = 1     # nose tip
_MP_MOUTH_LEFT  = 61    # left mouth corner
_MP_MOUTH_RIGHT = 291   # right mouth corner
_MP_CHIN        = 199
_MP_LEFT_EYE_EAR  = (33,  160, 158, 133, 153, 144)
_MP_RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)

_landmarker      = None        # cached FaceLandmarker instance
_landmarker_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal: lazy-init with double-check locking
# ---------------------------------------------------------------------------

def _get_detector():
    """Lazy-load and cache the MediaPipe FaceLandmarker (Tasks API)."""
    global _landmarker

    # Fast path – already initialised
    if _landmarker is not None:
        return _landmarker

    with _landmarker_lock:
        # Re-check inside lock (another thread may have initialised while waiting)
        if _landmarker is not None:
            return _landmarker

        try:
            import mediapipe as mp
            BaseOptions           = mp.tasks.BaseOptions
            FaceLandmarker        = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode     = mp.tasks.vision.RunningMode
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
            num_faces=2,                          # detect >1 so we can reject ambiguous frames
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
        return _landmarker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warmup() -> None:
    """Force-load the MediaPipe model before the door loop starts."""
    try:
        _get_detector()
        try:
            import mediapipe as mp
            logger.info("MediaPipe FaceLandmarker warmed up (mediapipe %s).", mp.__version__)
        except Exception:
            logger.info("MediaPipe FaceLandmarker warmed up.")
    except Exception as exc:
        logger.warning("Face detector warmup failed (non-fatal): %s", exc)


def release() -> None:
    """Release MediaPipe resources.  Call at application shutdown."""
    global _landmarker
    with _landmarker_lock:
        if _landmarker is not None:
            try:
                _landmarker.close()
            except Exception as exc:
                logger.debug("FaceLandmarker.close() raised: %s", exc)
            _landmarker = None
            logger.info("MediaPipe FaceLandmarker released.")


def detect_face(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Detect a single face in *frame* using MediaPipe FaceLandmarker.

    Returns
    -------
    dict | None
        Detection result with keys:
          - face_crop  : RGB np.ndarray, shape (160, 160, 3), uint8
          - bbox       : [x1, y1, x2, y2]  (pixel coords)
          - landmarks  : dict with named keypoints + EAR values
          - confidence : float (1.0 – MediaPipe Tasks API has no scalar score)
        {"error": "multiple_faces"} if > 1 face is present.
        None when no face detected or on any error.
    """
    # Validate input frame
    if frame is None:
        logger.debug("detect_face: received None frame.")
        return None
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        logger.debug("detect_face: invalid frame shape %s.", getattr(frame, "shape", "n/a"))
        return None
    if frame.size == 0:
        logger.debug("detect_face: empty frame.")
        return None

    try:
        import mediapipe as mp
    except ImportError as exc:
        logger.error("detect_face: mediapipe not available: %s", exc)
        return None

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return None

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except cv2.error as exc:
        logger.error("detect_face: cvtColor failed: %s", exc)
        return None

    try:
        landmarker = _get_detector()
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _landmarker_lock:
            result = landmarker.detect(mp_image)
    except Exception as exc:
        logger.error("detect_face: MediaPipe detection raised: %s", exc)
        return None

    if not result.face_landmarks:
        return None

    # Security: reject ambiguous multi-face frames
    if len(result.face_landmarks) > 1:
        return {"error": "multiple_faces", "count": len(result.face_landmarks)}

    face_lm = result.face_landmarks[0]   # list[NormalizedLandmark]
    total_lm = len(face_lm)

    def _px(lm_idx: int):
        if lm_idx >= total_lm:
            return (0, 0)
        lm = face_lm[lm_idx]
        return (
            int(max(0, min(w - 1, lm.x * w))),
            int(max(0, min(h - 1, lm.y * h))),
        )

    def _ear(indices):
        try:
            p = [np.array(_px(i), dtype=np.float32) for i in indices]
            vertical_1 = np.linalg.norm(p[1] - p[5])
            vertical_2 = np.linalg.norm(p[2] - p[4])
            horizontal = np.linalg.norm(p[0] - p[3])
            if horizontal < 1e-6:
                return 0.0
            return float((vertical_1 + vertical_2) / (2.0 * horizontal))
        except Exception:
            return 0.0

    landmarks = {
        "left_eye":     _px(_MP_LEFT_EYE),
        "right_eye":    _px(_MP_RIGHT_EYE),
        "nose":         _px(_MP_NOSE),
        "mouth_left":   _px(_MP_MOUTH_LEFT),
        "mouth_right":  _px(_MP_MOUTH_RIGHT),
        "chin":         _px(_MP_CHIN),
        "left_eye_ear":  _ear(_MP_LEFT_EYE_EAR),
        "right_eye_ear": _ear(_MP_RIGHT_EYE_EAR),
    }

    # Derive bounding box
    xs = [int(lm.x * w) for lm in face_lm]
    ys = [int(lm.y * h) for lm in face_lm]
    if not xs or not ys:
        return None

    pad = max(0, int(0.15 * (max(xs) - min(xs))))
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)

    if x2 <= x1 or y2 <= y1:
        logger.debug("detect_face: degenerate bounding box.")
        return None

    width = x2 - x1
    height = y2 - y1
    if width < 40 or height < 40:
        logger.debug("detect_face: face too small (%dx%d), rejecting.", width, height)
        return None

    face_region = rgb[y1:y2, x1:x2]
    if face_region.size == 0:
        logger.debug("detect_face: face_region is empty after crop.")
        return None

    # Eye-based similarity alignment (rotation + scale + translation) for
    # consistent pose normalization before embedding extraction.
    face_crop: Optional[np.ndarray] = None
    try:
        rel_left_eye = (int(landmarks["left_eye"][0] - x1), int(landmarks["left_eye"][1] - y1))
        rel_right_eye = (int(landmarks["right_eye"][0] - x1), int(landmarks["right_eye"][1] - y1))
        face_crop = _align_face_by_eyes(face_region, rel_left_eye, rel_right_eye, out_size=160)
    except Exception as exc:
        logger.debug("detect_face: eye alignment skipped: %s", exc)

    if face_crop is None:
        try:
            face_crop = cv2.resize(face_region, (160, 160))
        except cv2.error as exc:
            logger.error("detect_face: resize failed: %s", exc)
            return None

    if not _passes_quality_gates(face_crop):
        logger.debug("detect_face: quality gates failed; rejecting crop.")
        return None

    return {
        "face_crop":  face_crop,          # RGB (160, 160, 3)
        "bbox":       [x1, y1, x2, y2],
        "landmarks":  landmarks,
        "confidence": 1.0,
    }
