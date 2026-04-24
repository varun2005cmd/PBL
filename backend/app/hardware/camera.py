# =============================================================================
# app/hardware/camera.py  –  v3 (Ultimate Pi 5 Fix)
# =============================================================================

from __future__ import annotations
import logging
import os
import threading
import time
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CAMERA_INDEX    = int(os.environ.get("CAMERA_INDEX", "0"))
_CAPTURE_WIDTH   = int(os.environ.get("CAMERA_WIDTH", "640"))
_CAPTURE_HEIGHT  = int(os.environ.get("CAMERA_HEIGHT", "480"))
_CAPTURE_FPS     = int(os.environ.get("CAMERA_FPS", "15"))
_READ_RETRY      = 3
_FRAME_TIMEOUT_S = 3.0

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_cap: Optional[cv2.VideoCapture] = None
_cap_backend: str = "none"
_cap_lock = threading.RLock()

def _configure_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          _CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

def _probe_capture(cap: cv2.VideoCapture) -> bool:
    if not cap.isOpened():
        return False
    for _ in range(5): cap.grab()
    ok, frame = cap.read()
    return bool(ok and frame is not None)

def _open_camera_locked() -> Optional[cv2.VideoCapture]:
    global _cap, _cap_backend
    indices = [_CAMERA_INDEX, 0, 1, 2]
    
    for idx in indices:
        # Method 1: Standard
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                _configure_capture(cap)
                if _probe_capture(cap):
                    _cap = cap
                    _cap_backend = f"idx{idx}"
                    logger.info("Camera ready at index %d", idx)
                    return cap
                cap.release()
        except: pass

        # Method 2: V4L2 Force
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                _configure_capture(cap)
                if _probe_capture(cap):
                    _cap = cap
                    _cap_backend = f"v4l2-idx{idx}"
                    logger.info("Camera ready at v4l2-index %d", idx)
                    return cap
                cap.release()
        except: pass

    logger.error("No working camera found.")
    return None

def capture_frame() -> Optional[np.ndarray]:
    with _cap_lock:
        global _cap
        if _cap is None or not _cap.isOpened():
            _cap = _open_camera_locked()
        
        if _cap is None: return None
        
        _cap.grab()
        for _ in range(_READ_RETRY):
            ret, frame = _cap.read()
            if ret and frame is not None:
                return frame
        
        # Reset on failure
        _cap.release()
        _cap = None
        return None

def capture_frame_jpeg(quality: int = 85) -> Optional[bytes]:
    frame = capture_frame()
    if frame is None: return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else None

def wait_for_face(timeout: float = 10.0) -> Optional[np.ndarray]:
    from app.ml.face_detector import detect_face
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = capture_frame()
        if frame is not None:
            if detect_face(frame).get("face_crop") is not None:
                return frame
        time.sleep(0.3)
    return None

def release() -> None:
    with _cap_lock:
        global _cap
        if _cap:
            _cap.release()
            _cap = None

def status() -> dict:
    return {"opened": _cap is not None, "backend": _cap_backend}
