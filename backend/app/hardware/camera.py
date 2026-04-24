# =============================================================================
# app/hardware/camera.py  –  v4 (Ultimate Pi 5 Fix)
# =============================================================================

from __future__ import annotations
import logging
import os

# IMPORTANT: Set environment variables BEFORE importing cv2 to completely kill OBSENSOR
os.environ["OPENCV_VIDEOIO_EXCLUDE_LIST"] = "OBSENSOR"
os.environ["OPENCV_VIDEOIO_PRIORITY_LIST"] = "V4L2"

import cv2
import threading
import time
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Constants for configuration
_CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 480
_CAPTURE_FPS = 15

# ---------------------------------------------------------------------------
# Hardware State
# ---------------------------------------------------------------------------
_cap = None
_cap_lock = threading.Lock()
_last_frame = None

def _open_camera_locked():
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap

    # We ONLY try the index from the environment variable, and maybe the other common one.
    indices_to_try = [_CAMERA_INDEX]
    if _CAMERA_INDEX == 0: indices_to_try.append(1)
    elif _CAMERA_INDEX == 1: indices_to_try.append(0)

    for idx in indices_to_try:
        logger.info(f"Attempting to open camera at INDEX {idx} using strict V4L2...")
        
        # Pass INTEGER to avoid "can't capture by name" warnings
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, _CAPTURE_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Non-blocking grab check
            if cap.grab():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"SUCCESS! Camera fully operational at INDEX {idx}")
                    _cap = cap
                    return _cap
            
            logger.warning(f"Index {idx} opened but could not read frames. Releasing.")
            cap.release()
            
    logger.error("Camera completely failed to open. Please check the USB connection.")
    return None

def capture_frame() -> Optional[np.ndarray]:
    global _last_frame
    with _cap_lock:
        cap = _open_camera_locked()
        if cap is None:
            return None
            
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame, resetting camera...")
            cap.release()
            global _cap
            _cap = None
            return None
            
        _last_frame = frame.copy()
        return _last_frame

def capture_frame_jpeg(quality: int = 85) -> Optional[bytes]:
    frame = capture_frame()
    if frame is None: return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else None

def wait_for_face(timeout: float = 10.0, poll_interval: float = 0.3) -> Optional[np.ndarray]:
    from app.ml.face_detector import detect_face
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = capture_frame()
        if frame is not None:
            result = detect_face(frame)
            if result is not None and result.get("face_crop") is not None:
                return frame
        time.sleep(poll_interval)
    return None

def release() -> None:
    with _cap_lock:
        global _cap
        if _cap:
            _cap.release()
            _cap = None

def status() -> dict:
    return {"opened": _cap is not None, "backend": "V4L2"}
