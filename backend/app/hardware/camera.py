# =============================================================================
# app/hardware/camera.py
# USB Camera Capture Utility for Raspberry Pi 5
#
# Provides helpers for:
#   - Opening a USB camera (V4L2 via OpenCV)
#   - Capturing single frames for authentication
#   - Running a continuous capture loop (for the liveness challenge flow)
#
# The USB camera appears as /dev/video0 (or video1, etc.) on Linux.
# OpenCV's VideoCapture(0) is sufficient for most UVC-compatible USB cams.
#
# Optional: set CAMERA_INDEX env var to override the default device index.
# =============================================================================

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CAMERA_INDEX  = int(os.environ.get("CAMERA_INDEX", "0"))
_CAPTURE_WIDTH  = int(os.environ.get("CAMERA_WIDTH",  "1280"))
_CAPTURE_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))
_CAPTURE_FPS    = int(os.environ.get("CAMERA_FPS",    "30"))

# ---------------------------------------------------------------------------
# Module-level singleton capture object (kept open for performance)
# A threading lock prevents simultaneous reads from door loop and /camera/frame
# ---------------------------------------------------------------------------
import threading as _threading
_cap: Optional[cv2.VideoCapture] = None
_cap_lock = _threading.Lock()


def _get_cap() -> Optional[cv2.VideoCapture]:
    """
    Return a live VideoCapture instance, opening it if necessary.
    Returns None if the camera cannot be opened.
    """
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap

    logger.info(
        "Opening USB camera (index=%d, %dx%d @ %d fps).",
        _CAMERA_INDEX, _CAPTURE_WIDTH, _CAPTURE_HEIGHT, _CAPTURE_FPS,
    )
    # cv2.CAP_V4L2 only exists on Linux; fall back to auto-select on other OS
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = cv2.VideoCapture(_CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          _CAPTURE_FPS)
    # Reduce internal buffer to 1 frame to always get the latest image
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error(
            "Could not open camera at index %d. "
            "Check that the USB camera is connected and recognised as /dev/video%d.",
            _CAMERA_INDEX, _CAMERA_INDEX,
        )
        return None

    # Discard the first several frames so the sensor's auto-exposure and
    # auto-white-balance algorithms settle before we use any frame for ML.
    for _ in range(5):
        cap.grab()

    _cap = cap
    return _cap


def capture_frame() -> Optional[np.ndarray]:
    """
    Capture a single BGR frame from the USB camera.

    Returns
    -------
    np.ndarray | None
        BGR image array ready for the ML pipeline, or None on failure.
    """
    with _cap_lock:
        cap = _get_cap()
        if cap is None:
            return None

        # Discard one buffered frame to get the freshest image
        cap.grab()
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning("Camera read failed – attempting to reopen.")
            release()               # force re-open on next call
            return None

    return frame


def capture_frame_jpeg(quality: int = 85) -> Optional[bytes]:
    """
    Capture a frame and return it encoded as JPEG bytes.
    Useful for streaming to the frontend or storing as an attachment.
    """
    frame = capture_frame()
    if frame is None:
        return None
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if success else None


def wait_for_face(
    timeout: float = 10.0,
    poll_interval: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Poll the camera until a face is detected or *timeout* seconds elapse.

    poll_interval defaults to 0.3 s (≈3 fps face checks) to keep Pi CPU
    usage low during the idle waiting phase.

    Returns the first frame that contains a detectable face, or None.
    """
    from app.ml.face_detector import detect_face

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = capture_frame()
        if frame is not None and detect_face(frame) is not None:
            return frame
        time.sleep(poll_interval)

    logger.info("wait_for_face: timed out after %.1f s.", timeout)
    return None


def release() -> None:
    """Release the camera resource. Call at application shutdown or on error."""
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
        logger.info("Camera released.")
