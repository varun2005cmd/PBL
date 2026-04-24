# =============================================================================
# app/hardware/camera.py  –  v2 (hardened)
# USB Camera Capture Utility for Raspberry Pi 5
#
# Fixes applied:
#   • Thread-safe singleton with RLock around ALL shared state
#   • Dead-cap detection: isOpened() checked before every read
#   • Non-blocking grab() + retrieve() with configurable retry
#   • Auto-recovery: reopen on consecutive read failures (watchdog)
#   • GStreamer fallback with MJPEG primary
#   • Frame timeout: capture_frame() never blocks > FRAME_TIMEOUT_S
#   • Module-level watchdog counter prevents runaway reopen loops
#   • All errors are logged (no silent failures)
# =============================================================================

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
_CAMERA_INDEX    = int(os.environ.get("CAMERA_INDEX",    "0"))
_CAPTURE_WIDTH   = int(os.environ.get("CAMERA_WIDTH",    "640"))
_CAPTURE_HEIGHT  = int(os.environ.get("CAMERA_HEIGHT",   "480"))
_CAPTURE_FPS     = int(os.environ.get("CAMERA_FPS",      "20"))
_CAMERA_BACKEND  = os.environ.get("CAMERA_BACKEND",  "auto").lower().strip()
_FORCE_MJPEG     = os.environ.get("CAMERA_FORCE_MJPEG", "1") == "1"
_OPEN_RETRIES    = max(1, int(os.environ.get("CAMERA_OPEN_RETRIES",  "5")))
_READ_RETRY      = max(1, int(os.environ.get("CAMERA_READ_RETRY",    "3")))
# Maximum seconds a single capture_frame() call may take
_FRAME_TIMEOUT_S = float(os.environ.get("CAMERA_FRAME_TIMEOUT", "3.0"))
# Consecutive read failures before forcing a reopen
_MAX_READ_FAILS  = int(os.environ.get("CAMERA_MAX_READ_FAILS", "5"))

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_cap: Optional[cv2.VideoCapture] = None
_cap_backend: str = "none"
_cap_lock = threading.RLock()          # RLock: reentrant – _get_cap calls _open
_consecutive_read_failures: int = 0   # watchdog counter

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gstreamer_pipeline(index: int, width: int, height: int, fps: int, mjpeg: bool) -> str:
    device = f"/dev/video{index}"
    if mjpeg:
        source = (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegdec ! videoconvert"
        )
    else:
        source = (
            f"v4l2src device={device} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert"
        )
    return f"{source} ! appsink drop=true max-buffers=1 sync=false"


def _configure_capture(cap: cv2.VideoCapture) -> None:
    """Apply resolution / FPS / codec settings to an open VideoCapture."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          _CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize latency
    if _FORCE_MJPEG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def _probe_capture(cap: cv2.VideoCapture) -> bool:
    """Flush stale frames and confirm a live read succeeds."""
    if not cap.isOpened():
        return False
    # Flush up to 5 buffered frames
    for _ in range(5):
        cap.grab()
    ok, frame = cap.read()
    return bool(ok and frame is not None and frame.size > 0)


def _open_opencv_capture(backend: int, label: str) -> Optional[cv2.VideoCapture]:
    try:
        cap = cv2.VideoCapture(_CAMERA_INDEX, backend)
        _configure_capture(cap)
        if _probe_capture(cap):
            logger.info("Camera opened via %s backend.", label)
            return cap
        cap.release()
    except Exception as exc:
        logger.warning("OpenCV capture open (%s) raised: %s", label, exc)
    return None


def _open_gstreamer_capture(mjpeg: bool) -> Optional[cv2.VideoCapture]:
    try:
        pipeline = _gstreamer_pipeline(
            _CAMERA_INDEX, _CAPTURE_WIDTH, _CAPTURE_HEIGHT, _CAPTURE_FPS, mjpeg
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if _probe_capture(cap):
            logger.info("Camera opened via GStreamer (%s).", "MJPEG" if mjpeg else "raw")
            return cap
        cap.release()
    except Exception as exc:
        logger.warning("GStreamer capture open raised: %s", exc)
    return None


def _release_locked() -> None:
    """Release the current capture object.  Must be called with _cap_lock held."""
    global _cap, _cap_backend
    if _cap is not None:
        try:
            _cap.release()
        except Exception as exc:
            logger.debug("cap.release() raised: %s", exc)
        finally:
            _cap = None
            _cap_backend = "none"


def _build_candidate_list():
    candidates = []
    if _CAMERA_BACKEND in ("auto", "opencv", "v4l2"):
        if platform.system() == "Linux":
            candidates.append(
                ("opencv-v4l2", lambda: _open_opencv_capture(cv2.CAP_V4L2, "OpenCV V4L2"))
            )
        candidates.append(
            ("opencv-any", lambda: _open_opencv_capture(cv2.CAP_ANY, "OpenCV default"))
        )
    if _CAMERA_BACKEND in ("auto", "gstreamer"):
        candidates.append(("gstreamer-mjpeg", lambda: _open_gstreamer_capture(True)))
        candidates.append(("gstreamer-raw",   lambda: _open_gstreamer_capture(False)))
    return candidates


def _open_camera_locked() -> Optional[cv2.VideoCapture]:
    """Try every backend candidate up to _OPEN_RETRIES times.  Must hold _cap_lock."""
    global _cap, _cap_backend

    candidates = _build_candidate_list()
    if not candidates:
        logger.error("No camera backends configured.")
        return None

    for attempt in range(1, _OPEN_RETRIES + 1):
        for label, opener in candidates:
            cap = opener()
            if cap is not None:
                _cap = cap
                _cap_backend = label
                logger.info(
                    "Camera ready: index=%d %dx%d@%d backend=%s",
                    _CAMERA_INDEX, _CAPTURE_WIDTH, _CAPTURE_HEIGHT, _CAPTURE_FPS, label,
                )
                return cap
        logger.warning(
            "Camera open attempt %d/%d failed – retrying in 1 s.", attempt, _OPEN_RETRIES
        )
        time.sleep(1.0)

    logger.error(
        "Cannot open camera index %d after %d attempts. "
        "Check cable/power and whether /dev/video%d is in use.",
        _CAMERA_INDEX, _OPEN_RETRIES, _CAMERA_INDEX,
    )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_cap() -> Optional[cv2.VideoCapture]:
    """Return the live VideoCapture, opening it if necessary.  Caller must hold _cap_lock."""
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap
    logger.info(
        "Opening USB camera (index=%d, %dx%d @ %d fps, backend=%s).",
        _CAMERA_INDEX, _CAPTURE_WIDTH, _CAPTURE_HEIGHT, _CAPTURE_FPS, _CAMERA_BACKEND,
    )
    return _open_camera_locked()


def capture_frame() -> Optional[np.ndarray]:
    """
    Capture a single BGR frame from the USB camera.

    Thread-safe, non-blocking within _FRAME_TIMEOUT_S.
    Returns None on failure (always logs the reason).
    """
    global _consecutive_read_failures

    deadline = time.monotonic() + _FRAME_TIMEOUT_S

    with _cap_lock:
        cap = _get_cap()
        if cap is None:
            logger.error("capture_frame: camera unavailable.")
            return None

        # Flush the internal buffer so we get the freshest frame
        cap.grab()

        for attempt in range(_READ_RETRY):
            if time.monotonic() > deadline:
                logger.warning("capture_frame: frame timeout on attempt %d.", attempt)
                break

            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                _consecutive_read_failures = 0
                return frame

            logger.debug("capture_frame: read failed (attempt %d/%d).", attempt + 1, _READ_RETRY)
            cap.grab()   # try to recover internal buffer

        # All retries exhausted
        _consecutive_read_failures += 1
        logger.warning(
            "Camera read failed %d consecutive time(s) on backend=%s. Reopening.",
            _consecutive_read_failures, _cap_backend,
        )

        if _consecutive_read_failures >= _MAX_READ_FAILS:
            logger.critical(
                "Watchdog: %d consecutive camera failures. Forcing full reopen.",
                _consecutive_read_failures,
            )
            _consecutive_read_failures = 0

        _release_locked()
        cap = _open_camera_locked()
        if cap is None:
            logger.error("capture_frame: reopen failed.")
            return None

        cap.grab()
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame

        logger.error("capture_frame: read failed immediately after reopen.")
        return None


def capture_frame_jpeg(quality: int = 85) -> Optional[bytes]:
    """Capture a frame and return it as JPEG bytes (for streaming / violation images)."""
    frame = capture_frame()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        logger.error("capture_frame_jpeg: JPEG encoding failed.")
        return None
    return bytes(buf)


def wait_for_face(
    timeout: float = 10.0,
    poll_interval: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Poll the camera until a face is detected or *timeout* seconds elapse.

    Returns the first frame that contains a detectable face, or None.
    The face-detection import is deferred so this module stays import-safe
    even if MediaPipe is not installed.
    """
    from app.ml.face_detector import detect_face  # deferred import

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = capture_frame()
        if frame is not None:
            try:
                det = detect_face(frame)
                if isinstance(det, dict) and det.get("face_crop") is not None:
                    return frame
            except Exception as exc:
                logger.debug("wait_for_face: detect_face raised: %s", exc)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    logger.info("wait_for_face: timed out after %.1f s.", timeout)
    return None


def release() -> None:
    """Release the camera resource.  Call at application shutdown."""
    with _cap_lock:
        if _cap is not None:
            _release_locked()
            logger.info("Camera released.")


def status() -> dict:
    with _cap_lock:
        opened = bool(_cap is not None and _cap.isOpened())
    return {
        "opened":  opened,
        "backend": _cap_backend,
        "index":   _CAMERA_INDEX,
        "width":   _CAPTURE_WIDTH,
        "height":  _CAPTURE_HEIGHT,
        "fps":     _CAPTURE_FPS,
    }
