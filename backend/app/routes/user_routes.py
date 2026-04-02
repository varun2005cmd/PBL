"""
app/routes/user_routes.py
Flask Blueprint  User Management & Face-Recognition Authentication
=====================================================================

Endpoints
---------
GET  /users                  List all enrolled users.
POST /users                  Register a new user record (no face data yet).
DELETE /users/<id>           Remove a user and their embeddings.

GET  /logs                   Return all authentication logs (newest first).

POST /challenge              Issue a new liveness challenge token.
                             Response: {"challenge": "LEFT"|"RIGHT", "issued_at": float}

POST /authenticate           Main authentication endpoint.
                             Accepts: multipart form-data
                               - image : JPEG/PNG file (camera frame)
                               - challenge : "LEFT" or "RIGHT"
                               - issued_at : float (from /challenge response)
                             Response: {"status", "user", "liveness", "confidence", "detail"}

POST /enroll/<user_id>       Enrol a user by uploading 110 face images.
                             Accepts: multipart form-data, field "images" (multiple files)
                             Computes FaceNet embeddings and stores them on the User record.

POST /train                  (Re)train the SVM classifier on all enrolled users.
                             Call after adding/removing enrolled users.

POST /scan                   Legacy mock endpoint kept for backward-compatibility
                             with the existing UI.  Now uses the real ML pipeline.
"""

import time
import logging
from datetime import datetime

from flask import Blueprint, request, jsonify

from app.models       import db
from app.models.user  import User
from app.models.log   import AccessLog

logger = logging.getLogger(__name__)
user_bp = Blueprint("user_bp", __name__)


# =============================================================================
# Helper  actuate physical hardware after an auth result (best-effort)
# =============================================================================

def _trigger_hardware(result: dict) -> None:
    """
    Drive the servo and LCD based on the authentication result.
    Runs in a daemon thread so it never blocks the HTTP response.
    Silently ignored if hardware is unavailable (dev machine / CI).
    """
    import threading

    def _act():
        try:
            from app.hardware import servo, lcd
            if result["status"] == "granted":
                lcd.show_access_granted(result.get("user", ""))
                servo.unlock_door()
            else:
                detail = result.get("detail", "Denied")[:16]
                lcd.show_access_denied(detail)
        except Exception as exc:
            logger.debug("Hardware actuation skipped: %s", exc)

    t = threading.Thread(target=_act, daemon=True)
    t.start()


# =============================================================================
# Helper  load prototype embeddings from DB into a dict
# =============================================================================

def _load_prototype_embeddings() -> dict:
    """
    Return {username: [np.ndarray, ...]} for every active, enrolled user.
    Used by authenticate_user() and the SVM trainer.
    """
    users = User.query.filter_by(active=True, enrolled=True).all()
    return {u.name: u.get_np_embeddings() for u in users if u.get_np_embeddings()}


# =============================================================================
# User Management
# =============================================================================

@user_bp.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])


@user_bp.route("/users", methods=["POST"])
def add_user():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    if User.query.filter_by(name=name).first():
        return jsonify({"error": f"User '{name}' already exists"}), 409

    user = User(name=name)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User added", "user": user.to_dict()}), 201


@user_bp.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id: int):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": f"User {user_id} deleted."})


# =============================================================================
# Access Logs
# =============================================================================

@user_bp.route("/logs", methods=["GET"])
def get_logs():
    logs = AccessLog.query.order_by(AccessLog.id.desc()).all()
    return jsonify([l.to_dict() for l in logs])


# =============================================================================
# Liveness Challenge Token
# =============================================================================

@user_bp.route("/challenge", methods=["POST"])
def issue_challenge():
    """
    Issue a fresh liveness challenge.  The client must:
      1. Display the returned direction to the user.
      2. Capture a frame after the user performs the head turn.
      3. Send both the frame and the returned token fields to POST /authenticate.
    """
    from app.ml.liveness import generate_challenge
    token = generate_challenge()
    return jsonify(token)


# =============================================================================
# Main Authentication Endpoint
# =============================================================================

@user_bp.route("/authenticate", methods=["POST"])
def authenticate():
    """
    POST /authenticate
    ------------------
    Form-data fields:
      image      : image file (JPEG / PNG)
      challenge  : "LEFT" or "RIGHT"
      issued_at  : float (Unix timestamp from /challenge)

    Returns the standard auth result JSON.
    """
    # ---------- parse inputs ----------
    image_file  = request.files.get("image")
    challenge   = (request.form.get("challenge") or "").upper().strip()
    issued_at   = request.form.get("issued_at")

    if not image_file:
        return jsonify({"error": "No image provided"}), 400
    if challenge not in ("LEFT", "RIGHT"):
        return jsonify({"error": "challenge must be LEFT or RIGHT"}), 400
    try:
        issued_at = float(issued_at)
    except (TypeError, ValueError):
        return jsonify({"error": "issued_at must be a Unix timestamp float"}), 400

    # ---------- decode frame ----------
    from app.ml.pipeline import authenticate_user, decode_frame
    frame = decode_frame(image_file.read())
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    # ---------- run pipeline ----------
    prototypes = _load_prototype_embeddings()
    result     = authenticate_user(frame, challenge, issued_at, prototypes)

    # ---------- actuate hardware (non-blocking, best-effort) ----------
    _trigger_hardware(result)

    # ---------- persist audit log ----------
    log = AccessLog(
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status     = result["status"],
        user       = result["user"],
        liveness   = result["liveness"],
        confidence = result["confidence"],
        detail     = result.get("detail"),
        ip_address = request.remote_addr,
    )
    db.session.add(log)
    db.session.commit()

    http_status = 200 if result["status"] == "granted" else 403
    return jsonify(result), http_status


# =============================================================================
# Face Enrollment
# =============================================================================

@user_bp.route("/enroll/<int:user_id>", methods=["POST"])
def enroll_user(user_id: int):
    """
    POST /enroll/<user_id>
    ----------------------
    Upload between 3 and 10 JPEG/PNG face images for a user.
    Form-data field: "images" (allow multiple files with the same key).

    On success, stores computed FaceNet embeddings in the User record.
    Call POST /train afterwards to update the SVM classifier.
    """
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400
    if len(files) < 3:
        return jsonify({"error": "Minimum 3 images required for reliable enrollment"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 images per enrollment"}), 400

    from app.ml.pipeline import enroll_user as ml_enroll, decode_frame

    frames = []
    for f in files:
        frame = decode_frame(f.read())
        if frame is not None:
            frames.append(frame)

    if not frames:
        return jsonify({"error": "No decodable images in upload"}), 400

    enroll_result = ml_enroll(frames, user.name)

    if not enroll_result["ok"]:
        return jsonify({"error": enroll_result["message"]}), 422

    # Merge new embeddings with any existing ones
    existing = user.get_embeddings()
    user.set_embeddings(existing + enroll_result["embeddings"])
    db.session.commit()

    return jsonify({
        "message":  enroll_result["message"],
        "user":     user.to_dict(),
        "enrolled": enroll_result["count"],
    })


# =============================================================================
# SVM Training
# =============================================================================

@user_bp.route("/train", methods=["POST"])
def train_model():
    """
    POST /train
    -----------
    (Re)trains the SVM classifier on all currently enrolled users.
    Call this after every enrollment or user deletion.
    """
    from app.ml.recognizer import train_classifier

    prototypes = _load_prototype_embeddings()
    result     = train_classifier(prototypes)

    status = 200 if result["ok"] else 422
    return jsonify(result), status


# =============================================================================
# Legacy /scan  (backward-compatible  real ML, same response shape as mock)
# =============================================================================

@user_bp.route("/scan", methods=["POST"])
def scan_face():
    """
    Legacy endpoint kept for the existing UI dashboard.
    Accepts an image file upload and returns the same shape as the old mock:
      {"result": "Authorized"|"Unauthorized", "confidence": int}
    """
    image_file = request.files.get("image")
    if not image_file:
        # If called without a real image (e.g. from old UI), fall back to mock
        from app.mockml import mock_face_scan
        scan = mock_face_scan()
    else:
        from app.ml.pipeline import authenticate_user, decode_frame
        from app.ml.liveness import generate_challenge

        frame  = decode_frame(image_file.read())
        token  = generate_challenge()

        if frame is None:
            from app.mockml import mock_face_scan
            scan = mock_face_scan()
        else:
            prototypes = _load_prototype_embeddings()
            auth = authenticate_user(
                frame, token["challenge"], token["issued_at"], prototypes
            )
            scan = {
                "result":     "Authorized" if auth["status"] == "granted" else "Unauthorized",
                "confidence": int(auth["confidence"] * 100),
            }

    log = AccessLog(
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status     = "granted"  if scan["result"] == "Authorized" else "denied",
        user       = scan.get("user", "unknown"),
        liveness   = False,
        confidence = scan["confidence"] / 100.0,
        detail     = "Legacy /scan call",
        ip_address = request.remote_addr,
    )
    db.session.add(log)
    db.session.commit()

    if scan["result"] == "Unauthorized":
        return jsonify({"alert": "Unknown person detected!", **scan})

    return jsonify(scan)


# =============================================================================
# Toggle user active status
# =============================================================================

@user_bp.route("/users/<int:user_id>/toggle", methods=["PUT"])
def toggle_user(user_id: int):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    user.active = not user.active
    db.session.commit()
    # Return "status" as the string the frontend UserManagement page reads directly
    return jsonify({
        "success": True,
        "status":  "enabled" if user.active else "disabled",
        "active":  user.active,
        "user":    user.to_dict(),
    })


# =============================================================================
# Stats endpoint  (real data from DB for the dashboard)
# =============================================================================

@user_bp.route("/stats", methods=["GET"])
def get_stats():
    from sqlalchemy import func

    total_unlocks         = AccessLog.query.filter_by(status="granted").count()
    unauthorized_attempts = AccessLog.query.filter_by(status="denied").count()
    active_users          = User.query.filter_by(active=True, enrolled=True).count()

    total = total_unlocks + unauthorized_attempts
    detection_accuracy = round((total_unlocks / total * 100), 1) if total else 0.0

    liveness_pass = AccessLog.query.filter_by(liveness=True).count()
    liveness_pass_rate = round((liveness_pass / total * 100), 1) if total else 0.0

    avg_conf_row = db.session.query(
        func.avg(AccessLog.confidence)
    ).filter(AccessLog.status == "granted").scalar()
    avg_confidence = round(float(avg_conf_row or 0) * 100, 1)

    last_unauth = (
        AccessLog.query
        .filter_by(status="denied")
        .order_by(AccessLog.id.desc())
        .first()
    )

    return jsonify({
        "totalUnlocks":           total_unlocks,
        "unauthorizedAttempts":   unauthorized_attempts,
        "activeUsers":            active_users,
        "detectionAccuracy":      detection_accuracy,
        "livenessPassRate":       liveness_pass_rate,
        "avgConfidence":          avg_confidence,
        "lastUnauthorized":       last_unauth.timestamp if last_unauth else None,
    })


# =============================================================================
# Camera frame endpoint  (live snapshot from USB camera for the dashboard)
# =============================================================================

@user_bp.route("/camera/frame", methods=["GET"])
def camera_frame():
    """
    GET /camera/frame
    -----------------
    Returns the latest JPEG frame from the USB camera.
    The React dashboard can poll this endpoint to show a live feed.

    Response: JPEG image (Content-Type: image/jpeg)
    On failure: 503 JSON error.
    """
    from flask import Response
    from app.hardware.camera import capture_frame_jpeg

    jpeg = capture_frame_jpeg()
    if jpeg is None:
        return jsonify({"error": "Camera unavailable"}), 503

    return Response(jpeg, mimetype="image/jpeg")

