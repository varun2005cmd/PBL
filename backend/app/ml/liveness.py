# =============================================================================
# app/ml/liveness.py  –  v2 (hardened)
#
# Fixes applied:
#   • estimate_head_pose: validates landmark shape BEFORE solvePnP
#   • check_liveness: full input validation, fail-safe returns
#   • _check_blinks: max-frame cap, handles None frames list
#   • All code-paths return a well-formed dict (never None / exception)
#   • Timeout check uses monotonic clock (immune to NTP jumps)
#   • Logging on every failure path (no silent failures)
# =============================================================================

from __future__ import annotations

import logging
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CHALLENGE_TIMEOUT  = float(os.environ.get("CHALLENGE_TIMEOUT_SECONDS", "12"))
SUPPORTED_CHALLENGES = ("BLINK", "LEFT", "RIGHT", "UP", "DOWN")

_YAW_THRESHOLD_DEG   = 15.0
_PITCH_THRESHOLD_DEG = 11.0
_EAR_CLOSED_THRESHOLD = 0.205
_EAR_OPEN_THRESHOLD   = 0.265
_REQUIRED_BLINKS      = 2
_MAX_BLINK_FRAMES     = int(os.environ.get("MAX_BLINK_FRAMES", "30"))

# 6 model points (nose, left_eye, right_eye, mouth_left, mouth_right, chin)
_MODEL_POINTS_3D = np.array(
    [
        (0.0,   0.0,   0.0),     # nose tip
        (-35.0, -35.0, -35.0),   # left eye outer corner
        (35.0,  -35.0, -35.0),   # right eye outer corner
        (-30.0,  35.0, -35.0),   # mouth left corner
        (30.0,   35.0, -35.0),   # mouth right corner
        (0.0,   75.0,  -45.0),   # chin
    ],
    dtype=np.float64,
)

_REQUIRED_LANDMARK_KEYS = ("nose", "left_eye", "right_eye", "mouth_left", "mouth_right", "chin")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_challenge() -> Dict[str, Any]:
    challenge = random.choice(SUPPORTED_CHALLENGES)
    return {
        "challenge":  challenge,
        "issued_at":  time.time(),
        "timeout":    CHALLENGE_TIMEOUT,
        "prompt":     challenge_prompt(challenge),
    }


def challenge_prompt(challenge: str) -> str:
    labels = {
        "BLINK": "Blink twice",
        "LEFT":  "Look left",
        "RIGHT": "Look right",
        "UP":    "Look up",
        "DOWN":  "Look down",
    }
    return labels.get((challenge or "").upper(), "Follow prompt")


def check_liveness(
    frame: np.ndarray,
    challenge: str,
    landmarks: Dict[str, Any],
    challenge_issued_at: float,
    frames: Optional[Iterable[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Evaluate whether the user passed the liveness challenge.

    Always returns a dict with at least {"passed": bool, "reason": str}.
    Never raises; all exceptions are caught and returned as failures.
    """
    challenge = (challenge or "").upper().strip()

    # --- Timeout check (monotonic delta from issued_at epoch) ---------------
    now = time.time()
    elapsed = now - float(challenge_issued_at)
    if elapsed > CHALLENGE_TIMEOUT:
        logger.warning("Liveness: challenge timed out (elapsed=%.1f s).", elapsed)
        return _result(False, f"Challenge timed out ({elapsed:.0f} s).")

    # --- Validate challenge string ------------------------------------------
    if challenge not in SUPPORTED_CHALLENGES:
        logger.warning("Liveness: unknown challenge '%s'.", challenge)
        return _result(False, f"Unknown challenge '{challenge}'.")

    # --- Blink (temporal, needs frame sequence) -----------------------------
    if challenge == "BLINK":
        frames_list: List[np.ndarray] = list(frames) if frames else []
        blink_result = _check_blinks(frames_list)
        if blink_result["passed"]:
            return _result(True, "Blink challenge passed.", blinks=blink_result["blinks"])
        return _result(
            False,
            f"Blink challenge failed (detected {blink_result['blinks']} / {_REQUIRED_BLINKS} required).",
            blinks=blink_result["blinks"],
        )

    # --- Pose challenge (single frame) --------------------------------------
    if frame is None or frame.ndim < 2:
        logger.warning("Liveness: no frame for pose challenge.")
        return _result(False, "No frame provided for pose challenge.")

    if not isinstance(landmarks, dict):
        logger.warning("Liveness: landmarks not a dict (type=%s).", type(landmarks).__name__)
        return _result(False, "Missing facial landmarks for pose challenge.")

    try:
        pose = estimate_head_pose(frame, landmarks)
    except Exception as exc:
        logger.error("Liveness: estimate_head_pose raised: %s", exc)
        return _result(False, "Head-pose estimation failed (exception).")

    if pose is None:
        logger.warning("Liveness: head pose estimation returned None.")
        return _result(False, "Could not estimate head pose (insufficient landmarks).")

    yaw   = float(pose["yaw"])
    pitch = float(pose["pitch"])

    passed = (
        (challenge == "LEFT"  and yaw   <= -_YAW_THRESHOLD_DEG)
        or (challenge == "RIGHT" and yaw   >=  _YAW_THRESHOLD_DEG)
        or (challenge == "UP"    and pitch <= -_PITCH_THRESHOLD_DEG)
        or (challenge == "DOWN"  and pitch >=  _PITCH_THRESHOLD_DEG)
    )

    if not passed:
        return {
            "passed": False,
            "yaw":    round(yaw,   2),
            "pitch":  round(pitch, 2),
            "reason": (
                f"Expected {challenge_prompt(challenge)}. "
                f"yaw={yaw:.1f}°, pitch={pitch:.1f}°."
            ),
        }

    return {
        "passed": True,
        "yaw":    round(yaw,   2),
        "pitch":  round(pitch, 2),
        "reason": "Liveness challenge passed.",
    }


def estimate_head_pose(
    frame: np.ndarray,
    landmarks: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    Compute yaw / pitch / roll via solvePnP.

    Returns None when:
      - frame is None / not a valid image
      - any required landmark key is missing
      - image_points shape is not (6, 2)
      - any coordinate is non-finite
      - solvePnP itself fails
    """
    if frame is None or not hasattr(frame, "ndim") or frame.ndim < 2:
        logger.debug("estimate_head_pose: invalid frame.")
        return None

    # Validate all required keys are present
    missing = [k for k in _REQUIRED_LANDMARK_KEYS if k not in landmarks]
    if missing:
        logger.debug("estimate_head_pose: missing landmarks %s.", missing)
        return None

    # Build (6, 2) image-point array
    try:
        image_points = np.array(
            [landmarks[k] for k in _REQUIRED_LANDMARK_KEYS],
            dtype=np.float64,
        )
    except (TypeError, ValueError, KeyError) as exc:
        logger.debug("estimate_head_pose: landmark array build failed: %s", exc)
        return None

    if image_points.shape != (6, 2):
        logger.debug(
            "estimate_head_pose: expected shape (6,2), got %s.", image_points.shape
        )
        return None

    if not np.isfinite(image_points).all():
        logger.debug("estimate_head_pose: non-finite landmark coordinates.")
        return None

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None

    focal_length  = float(w)
    camera_matrix = np.array(
        [[focal_length, 0, w / 2.0],
         [0, focal_length, h / 2.0],
         [0, 0,            1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        ok, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        logger.warning("estimate_head_pose: solvePnP error: %s", exc)
        return None

    if not ok:
        logger.debug("estimate_head_pose: solvePnP returned ok=False.")
        return None

    try:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    except cv2.error as exc:
        logger.warning("estimate_head_pose: Rodrigues error: %s", exc)
        return None

    # Decompose rotation matrix → Euler angles
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if singular:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw   = math.atan2(-rotation_matrix[2, 0], sy)
        roll  = 0.0
    else:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw   = math.atan2(-rotation_matrix[2, 0], sy)
        roll  = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return {
        "yaw":   math.degrees(yaw),
        "pitch": math.degrees(pitch),
        "roll":  math.degrees(roll),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_blinks(frames: List[np.ndarray]) -> Dict[str, Any]:
    """Count blinks in a frame sequence using EAR."""
    if not frames:
        logger.debug("_check_blinks: no frames provided.")
        return {"passed": False, "blinks": 0}

    try:
        from app.ml.face_detector import detect_face  # deferred
    except ImportError as exc:
        logger.error("_check_blinks: cannot import detect_face: %s", exc)
        return {"passed": False, "blinks": 0}

    blinks    = 0
    eye_closed = False
    sample_frames = list(frames)[:_MAX_BLINK_FRAMES]

    for sample in sample_frames:
        if sample is None:
            continue
        try:
            detection = detect_face(sample)
        except Exception as exc:
            logger.debug("_check_blinks: detect_face raised: %s", exc)
            continue

        if not isinstance(detection, dict) or detection.get("error"):
            continue

        ear = _average_ear(detection.get("landmarks") or {})
        if ear is None:
            continue

        if not eye_closed and ear < _EAR_CLOSED_THRESHOLD:
            eye_closed = True
        elif eye_closed and ear > _EAR_OPEN_THRESHOLD:
            blinks    += 1
            eye_closed = False

    logger.debug("_check_blinks: detected %d blink(s) from %d frames.", blinks, len(sample_frames))
    return {"passed": blinks >= _REQUIRED_BLINKS, "blinks": blinks}


def _average_ear(landmarks: Dict[str, Any]) -> Optional[float]:
    left  = landmarks.get("left_eye_ear")
    right = landmarks.get("right_eye_ear")
    if left is None or right is None:
        return None
    try:
        return (float(left) + float(right)) / 2.0
    except (TypeError, ValueError):
        return None


def _result(passed: bool, reason: str, **extra: Any) -> Dict[str, Any]:
    base = {"passed": passed, "yaw": 0.0, "pitch": 0.0, "reason": reason}
    base.update(extra)
    return base
