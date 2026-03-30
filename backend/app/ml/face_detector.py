# =============================================================================
# app/ml/face_detector.py
# Face Detection using MTCNN
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

import numpy as np
import cv2
from typing import Optional, Dict, Any

# MTCNN is imported lazily so the module can be imported without
# GPU-heavy initialisation at startup; the detector is cached after first use.
_detector = None


def _get_detector():
    """Lazy-load and cache the MTCNN detector (CPU-only)."""
    global _detector
    if _detector is None:
        try:
            from mtcnn import MTCNN  # pip install mtcnn
            _detector = MTCNN()
        except ImportError as exc:
            raise ImportError(
                "MTCNN is not installed. Run: pip install mtcnn tensorflow"
            ) from exc
    return _detector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_face(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Detect a single face in *frame* using MTCNN.

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
          - landmarks  : dict with 5 named keypoints
          - confidence : float [0, 1]
        Returns None when no face / low-confidence detection.
    """
    # MTCNN expects RGB; OpenCV gives BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Downscale for speed on Raspberry Pi before detection
    scale = _compute_scale(rgb, max_dim=640)
    if scale < 1.0:
        small = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
    else:
        small = rgb
        scale = 1.0

    detector = _get_detector()
    results = detector.detect_faces(small)

    if not results:
        return None

    # Pick the detection with the highest confidence
    best = max(results, key=lambda r: r["confidence"])

    if best["confidence"] < 0.90:
        return None

    # Re-scale bbox / keypoints back to original resolution
    x, y, w, h = [int(v / scale) for v in best["box"]]
    # MTCNN box can have negative x/y — clamp
    x, y = max(0, x), max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    kp = best["keypoints"]
    landmarks = {
        "left_eye":    _rescale_pt(kp["left_eye"],    scale),
        "right_eye":   _rescale_pt(kp["right_eye"],   scale),
        "nose":        _rescale_pt(kp["nose"],         scale),
        "mouth_left":  _rescale_pt(kp["mouth_left"],  scale),
        "mouth_right": _rescale_pt(kp["mouth_right"], scale),
    }

    # Crop and resize to 160×160 (FaceNet input size)
    face_region = rgb[y:y2, x:x2]
    if face_region.size == 0:
        return None
    face_crop = cv2.resize(face_region, (160, 160))

    return {
        "face_crop":  face_crop,          # RGB (160, 160, 3)
        "bbox":       [x, y, x2, y2],
        "landmarks":  landmarks,
        "confidence": float(best["confidence"]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_scale(image: np.ndarray, max_dim: int) -> float:
    """Return a scale factor so the longest side fits within *max_dim*."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return 1.0
    return max_dim / longest


def _rescale_pt(pt: tuple, scale: float) -> tuple:
    """Inverse-scale a (x, y) keypoint from the downscaled space."""
    return (int(pt[0] / scale), int(pt[1] / scale))
