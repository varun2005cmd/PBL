# =============================================================================
# app/ml/liveness.py
# Liveness Detection  Active Challenge-Response (HEAD TURN)
#
# Strategy:
#   - Use OpenCV solvePnP with a generic 3-D face model to estimate head pose.
#   - Extract the YAW angle from the rotation vector.
#   - Challenge is either "LEFT" or "RIGHT".
#       LEFT   user must turn head so yaw < -20 degrees
#       RIGHT  user must turn head so yaw > +20 degrees
#   - A time window is enforced by the caller passing in
#     `challenge_issued_at` (Unix timestamp); challenge fails if
#     time.time() - challenge_issued_at > CHALLENGE_TIMEOUT (currently 15 s).
# =============================================================================

import time
import math
from typing import Optional, Dict, Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Generic 3-D face model landmarks (based on a neutral-pose average face).
# Order matches MTCNN keypoint order:
#   [left_eye, right_eye, nose, mouth_left, mouth_right]
# Coordinates are in an arbitrary metric unit (centimetres work well).
# ---------------------------------------------------------------------------
_MODEL_POINTS_3D = np.array(
    [
        # Nose tip
        (0.0,   0.0,    0.0),
        # Chin  (approximated below nose)
        (0.0,  -63.6,  -12.5),
        # Left eye corner (from subject's perspective = image-right)
        (-43.3,  32.7,  -26.0),
        # Right eye corner
        (43.3,   32.7,  -26.0),
        # Left mouth corner
        (-28.9, -28.9,  -24.1),
        # Right mouth corner
        (28.9,  -28.9,  -24.1),
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

YAW_LEFT_THRESHOLD  = -20.0   # degrees
YAW_RIGHT_THRESHOLD = +20.0   # degrees
CHALLENGE_TIMEOUT   =  15.0   # seconds (covers Pi ML processing time ~8-12 s)


def check_liveness(
    frame: np.ndarray,
    challenge: str,
    landmarks: Dict[str, tuple],
    challenge_issued_at: float,
) -> Dict[str, Any]:
    """
    Verify whether the user has performed the required head-turn challenge
    within the allowed time window.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from OpenCV (used only to read image dimensions for
        camera-matrix estimation).
    challenge : str
        Either ``"LEFT"`` or ``"RIGHT"`` (case-insensitive).
    landmarks : dict
        Five facial keypoints as returned by ``detect_face()``:
        ``left_eye``, ``right_eye``, ``nose``, ``mouth_left``, ``mouth_right``.
    challenge_issued_at : float
        ``time.time()`` value recorded when the challenge was issued.

    Returns
    -------
    dict
        ``{"passed": bool, "yaw": float, "reason": str}``
    """
    challenge = challenge.upper().strip()
    if challenge not in ("LEFT", "RIGHT"):
        return {"passed": False, "yaw": 0.0, "reason": "Invalid challenge direction."}

    # ------------------------------------------------------------------
    # 1. Time-window check
    # ------------------------------------------------------------------
    elapsed = time.time() - challenge_issued_at
    if elapsed > CHALLENGE_TIMEOUT:
        return {
            "passed": False,
            "yaw":    0.0,
            "reason": f"Challenge timed out after {elapsed:.1f}s (limit {CHALLENGE_TIMEOUT}s).",
        }

    # ------------------------------------------------------------------
    # 2. Build camera intrinsic matrix from image size (no calibration data)
    # ------------------------------------------------------------------
    h, w = frame.shape[:2]
    focal_length = w  # rough approximation
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1         ],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no lens distortion

    # ------------------------------------------------------------------
    # 3. Map 2-D landmarks to the 3-D model point order
    #    Order: nose, chin(approx), left_eye, right_eye, mouth_left, mouth_right
    # ------------------------------------------------------------------
    try:
        image_points_2d = np.array(
            [
                landmarks["nose"],
                _approx_chin(landmarks),       # derived from other landmarks
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["mouth_left"],
                landmarks["mouth_right"],
            ],
            dtype=np.float64,
        )
    except KeyError as exc:
        return {"passed": False, "yaw": 0.0, "reason": f"Missing landmark: {exc}"}

    # ------------------------------------------------------------------
    # 4. Solve PnP
    # ------------------------------------------------------------------
    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return {"passed": False, "yaw": 0.0, "reason": "PnP solver failed."}

    # ------------------------------------------------------------------
    # 5. Convert rotation vector  Euler angles, extract yaw
    # ------------------------------------------------------------------
    yaw_deg = _rotation_vector_to_yaw(rvec)

    # ------------------------------------------------------------------
    # 6. Evaluate challenge
    # ------------------------------------------------------------------
    # In OpenCV camera frame: turning head LEFT  → yaw > +threshold
    #                          turning head RIGHT → yaw < -threshold
    # If your setup produces the opposite sign, negate yaw_deg here.
    if challenge == "LEFT"  and yaw_deg > abs(YAW_LEFT_THRESHOLD):
        return {"passed": True,  "yaw": yaw_deg, "reason": "Liveness confirmed (left turn)."}
    if challenge == "RIGHT" and yaw_deg < -abs(YAW_RIGHT_THRESHOLD):
        return {"passed": True,  "yaw": yaw_deg, "reason": "Liveness confirmed (right turn)."}

    required = (
        f"> +{abs(YAW_LEFT_THRESHOLD):.0f}°" if challenge == "LEFT"
        else f"< -{abs(YAW_RIGHT_THRESHOLD):.0f}°"
    )
    return {
        "passed": False,
        "yaw":    yaw_deg,
        "reason": (
            f"Insufficient head turn for '{challenge}' challenge. "
            f"Yaw = {yaw_deg:.1f}° (required {required})."
        ),
    }


def generate_challenge() -> Dict[str, Any]:
    """
    Randomly issue a liveness challenge.

    Returns
    -------
    dict
        ``{"challenge": "LEFT"|"RIGHT", "issued_at": float}``
    """
    import random
    direction = random.choice(["LEFT", "RIGHT"])
    return {"challenge": direction, "issued_at": time.time()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _approx_chin(landmarks: dict) -> tuple:
    """
    Approximate chin position from mouth corners (below mouth midpoint).
    This avoids needing a separate chin detector.
    """
    mx = (landmarks["mouth_left"][0] + landmarks["mouth_right"][0]) / 2.0
    my = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2.0
    # Estimate chin ~40 % of inter-eye distance further down
    eye_dist = abs(landmarks["right_eye"][0] - landmarks["left_eye"][0])
    cy = my + eye_dist * 0.4
    return (mx, cy)


def _rotation_vector_to_yaw(rvec: np.ndarray) -> float:
    """
    Convert a Rodrigues rotation vector to a yaw (heading) angle in degrees.

    Convention used here:
      Positive yaw  = subject's nose pointing to image-RIGHT  (LEFT challenge)
      Negative yaw  = subject's nose pointing to image-LEFT   (RIGHT challenge)

    Note: cv2.RQDecomp3x3 returns (rx, ry, rz) where ry is the Y-axis
    (yaw) rotation in degrees.  OpenCV's camera frame has +Y pointing DOWN,
    so turning head RIGHT yields a NEGATIVE ry.  The thresholds at the call
    site must match this sign:
        LEFT  challenge  -> yaw > +YAW_LEFT_THRESHOLD  (absolute, see below)
        RIGHT challenge  -> yaw < -YAW_RIGHT_THRESHOLD
    Both thresholds are set symmetrically to ±20°.

    If your camera produces the opposite sign, swap the two threshold checks
    in check_liveness() or negate the returned value here.
    """
    rot_mat, _ = cv2.Rodrigues(rvec)
    # RQDecomp3x3 returns (angles_deg, Qx, Qy, Qz, Qx*Qy, Qx*Qy*Qz)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    # angles[0]=pitch, angles[1]=yaw, angles[2]=roll  (all in degrees)
    yaw_deg = angles[1]
    return float(yaw_deg)
