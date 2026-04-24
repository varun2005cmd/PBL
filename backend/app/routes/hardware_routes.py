# =============================================================================
# app/routes/hardware_routes.py  –  v2 (hardened)
#
# Fixes applied:
#   • Removed redundant local state (_door_status)
#   • Uses hardware modules (servo.py, camera.py) as Single Source of Truth
#   • Added try/except guards for all hardware interaction
#   • Simplified status logic to return REAL hardware availability
# =============================================================================

import logging
import os
import time
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)
hardware_bp = Blueprint("hardware_bp", __name__)

_API_KEY = os.environ.get("API_KEY")

def _authorised() -> bool:
    if not _API_KEY:
        return True
    return request.headers.get("X-API-Key") == _API_KEY

def _reject_unauthorised():
    return jsonify({
        "success": False,
        "error": "unauthorised",
        "message": "Missing or invalid API key",
    }), 401


@hardware_bp.route("/hardware/status", methods=["GET"])
def get_status():
    """Return the real-time door lock and sensor status."""
    if not _authorised():
        return _reject_unauthorised()
        
    try:
        from app.hardware import camera, servo, lcd
        
        # Query REAL hardware state from the modules
        srv_status = servo.status()
        cam_status = camera.status()
        lcd_status = lcd.status()
        
        return jsonify({
            "status":    "unlocked" if srv_status.get("unlocked") else "locked",
            "camera":    "connected" if cam_status.get("opened") else "disconnected",
            "servo":     "connected" if srv_status.get("available") else "disconnected",
            "lcd":       "connected" if lcd_status.get("available") else "disconnected",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as exc:
        logger.error("GET /hardware/status failed: %s", exc)
        return jsonify({"error": "Failed to query hardware status"}), 500


@hardware_bp.route("/hardware/lock", methods=["POST"])
def lock_door():
    """Lock the door via servo motor."""
    if not _authorised():
        return _reject_unauthorised()
    try:
        from app.hardware import servo
        servo.lock_door()
        return jsonify({"success": True, "status": "locked", "message": "Door locked"})
    except Exception as exc:
        logger.warning("REST /hardware/lock failed: %s", exc)
        return jsonify({"success": False, "message": "Hardware communication error"}), 503


@hardware_bp.route("/hardware/unlock", methods=["POST"])
def unlock_door():
    """Unlock the door via servo motor."""
    if not _authorised():
        return _reject_unauthorised()
    try:
        from app.hardware import servo
        servo.unlock_door()
        return jsonify({"success": True, "status": "unlocked", "message": "Door unlocked"})
    except Exception as exc:
        logger.warning("REST /hardware/unlock failed: %s", exc)
        return jsonify({"success": False, "message": "Hardware communication error"}), 503
