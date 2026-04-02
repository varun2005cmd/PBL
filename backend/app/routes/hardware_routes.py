"""
app/routes/hardware_routes.py
Flask Blueprint  Hardware Control (Servo, LCD, Status)
=====================================================

Endpoints
---------
GET  /hardware/status        Get current door lock status.
POST /hardware/lock          Lock the door (servo).
POST /hardware/unlock        Unlock the door (servo).
"""

import logging
import time
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)
hardware_bp = Blueprint("hardware_bp", __name__)

# Internal state — updated by both REST calls and the auto-relock timer
_door_status = "locked"
_lock_time: float = 0.0  # time.time() when auto-relock is expected


def _compute_status() -> str:
    """
    Return the real current status by checking whether the auto-relock
    timer has elapsed since the last unlock.  This means the status
    is always accurate even between REST calls.
    """
    global _door_status, _lock_time
    if _door_status == "unlocked" and _lock_time > 0 and time.time() >= _lock_time:
        _door_status = "locked"
        _lock_time = 0.0
    return _door_status


@hardware_bp.route("/hardware/status", methods=["GET"])
def get_status():
    """Return the real-time door lock status."""
    status = _compute_status()
    try:
        from app.hardware import camera as cam_mod, servo as srv_mod, lcd as lcd_mod
        cam_ok   = cam_mod._cap is not None and cam_mod._cap.isOpened()
        servo_ok = srv_mod._gpio_ok
        lcd_ok   = lcd_mod._lcd_ok
    except Exception:
        cam_ok = servo_ok = lcd_ok = False
    return jsonify({
        "status":    status,
        "camera":   "connected" if cam_ok   else "disconnected",
        "servo":    "connected" if servo_ok else "disconnected",
        "lcd":      "connected" if lcd_ok   else "disconnected",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@hardware_bp.route("/hardware/lock", methods=["POST"])
def lock_door():
    """Lock the door via servo motor."""
    global _door_status, _lock_time
    try:
        from app.hardware import servo
        servo.lock_door()
        _door_status = "locked"
        _lock_time = 0.0
        return jsonify({"success": True, "status": "locked", "message": "Door locked"})
    except Exception as exc:
        logger.warning("Lock failed: %s", exc)
        return jsonify({"success": False, "status": _door_status, "message": str(exc)}), 500


@hardware_bp.route("/hardware/unlock", methods=["POST"])
def unlock_door():
    """Unlock the door via servo motor."""
    global _door_status, _lock_time
    from app.hardware.servo import _UNLOCK_DURATION_S
    try:
        from app.hardware import servo
        servo.unlock_door()
        _door_status = "unlocked"
        _lock_time = time.time() + _UNLOCK_DURATION_S   # track when auto-relock fires
        return jsonify({"success": True, "status": "unlocked", "message": "Door unlocked"})
    except Exception as exc:
        logger.warning("Unlock failed: %s", exc)
        return jsonify({"success": False, "status": _door_status, "message": str(exc)}), 500
