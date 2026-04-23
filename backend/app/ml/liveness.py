from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

CHALLENGE_TIMEOUT = 12.0
SUPPORTED_CHALLENGES = ("BLINK", "LEFT", "RIGHT", "UP", "DOWN")

_YAW_THRESHOLD_DEG = 18.0
_PITCH_THRESHOLD_DEG = 13.0
_EAR_CLOSED_THRESHOLD = 0.21
_EAR_OPEN_THRESHOLD = 0.26
_REQUIRED_BLINKS = 2


def generate_challenge() -> Dict[str, Any]:
    challenge = random.choice(SUPPORTED_CHALLENGES)
    return {
        "challenge": challenge,
        "issued_at": time.time(),
        "timeout": CHALLENGE_TIMEOUT,
        "prompt": challenge_prompt(challenge),
    }


def challenge_prompt(challenge: str) -> str:
    labels = {
        "BLINK": "Blink twice",
        "LEFT": "Look left",
        "RIGHT": "Look right",
        "UP": "Look up",
        "DOWN": "Look down",
    }
    return labels.get((challenge or "").upper(), "Follow prompt")


def check_liveness(
    frame: np.ndarray,
    challenge: str,
    landmarks: Dict[str, Any],
    challenge_issued_at: float,
    frames: Optional[Iterable[np.ndarray]] = None,
) -> Dict[str, Any]:
    challenge = (challenge or "").upper().strip()
    elapsed = time.time() - float(challenge_issued_at)
    if elapsed > CHALLENGE_TIMEOUT:
        return _result(False, "Challenge timed out.", landmarks)

    if challenge == "BLINK":
        blink_result = _check_blinks(frames)
        if blink_result["passed"]:
            return _result(True, "Blink challenge passed.", landmarks, blinks=blink_result["blinks"])
        return _result(False, "Blink challenge failed. Blink twice within the timeout.", landmarks, blinks=blink_result["blinks"])

    pose = estimate_head_pose(frame, landmarks)
    if pose is None:
        return _result(False, "Could not estimate head pose.", landmarks)

    yaw = pose["yaw"]
    pitch = pose["pitch"]
    passed = (
        (challenge == "LEFT" and yaw <= -_YAW_THRESHOLD_DEG)
        or (challenge == "RIGHT" and yaw >= _YAW_THRESHOLD_DEG)
        or (challenge == "UP" and pitch <= -_PITCH_THRESHOLD_DEG)
        or (challenge == "DOWN" and pitch >= _PITCH_THRESHOLD_DEG)
    )
    if not passed:
        return {
            "passed": False,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "reason": f"Expected {challenge_prompt(challenge)}. yaw={yaw:.1f}, pitch={pitch:.1f}.",
        }

    return {
        "passed": True,
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "reason": "Liveness challenge passed.",
    }


def estimate_head_pose(frame: np.ndarray, landmarks: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        image_points = np.array(
            [
                landmarks["nose"],
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["mouth_left"],
                landmarks["mouth_right"],
            ],
            dtype=np.float64,
        )
    except KeyError:
        return None

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-35.0, -35.0, -35.0),
            (35.0, -35.0, -35.0),
            (-30.0, 35.0, -35.0),
            (30.0, 35.0, -35.0),
        ],
        dtype=np.float64,
    )

    h, w = frame.shape[:2]
    focal_length = float(w)
    camera_matrix = np.array(
        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if singular:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return {
        "yaw": math.degrees(yaw),
        "pitch": math.degrees(pitch),
        "roll": math.degrees(roll),
    }


def _check_blinks(frames: Optional[Iterable[np.ndarray]]) -> Dict[str, Any]:
    if not frames:
        return {"passed": False, "blinks": 0}

    from app.ml.face_detector import detect_face

    blinks = 0
    eye_closed = False
    for sample in frames:
        detection = detect_face(sample)
        if not isinstance(detection, dict) or detection.get("error"):
            continue
        ear = _average_ear(detection.get("landmarks", {}))
        if ear is None:
            continue
        if not eye_closed and ear < _EAR_CLOSED_THRESHOLD:
            eye_closed = True
        elif eye_closed and ear > _EAR_OPEN_THRESHOLD:
            blinks += 1
            eye_closed = False

    return {"passed": blinks >= _REQUIRED_BLINKS, "blinks": blinks}


def _average_ear(landmarks: Dict[str, Any]) -> Optional[float]:
    left = landmarks.get("left_eye_ear")
    right = landmarks.get("right_eye_ear")
    if left is None or right is None:
        return None
    return (float(left) + float(right)) / 2.0


def _result(
    passed: bool,
    reason: str,
    landmarks: Dict[str, Any],
    **extra: Any,
) -> Dict[str, Any]:
    base = {"passed": passed, "yaw": 0.0, "pitch": 0.0, "reason": reason}
    base.update(extra)
    return base
