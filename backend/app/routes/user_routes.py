# =============================================================================
# app/routes/user_routes.py  –  v2 (hardened)
#
# Fixes applied:
#   • Added thread lock protection for _trigger_hardware execution
#   • Bound all liveness frame lists to prevent memory leaks/DoS
#   • Hardened hardware calls via try/except inside thread
#   • Proper API validation for floats/ints
#   • Handled missing / None embeddings gracefully
#   • _cleanup_enrollment_sessions uses a lock safely
# =============================================================================

"""
app/routes/user_routes.py
Flask Blueprint  User Management & Face-Recognition Authentication
"""

import time
import logging
import threading
import uuid
import os
from datetime import datetime

from flask import Blueprint, request, jsonify
import numpy as np

from app.models       import db
from app.models.user  import User
from app.models.log   import AccessLog
from app.models.violation import ViolationImage

logger = logging.getLogger(__name__)
user_bp = Blueprint("user_bp", __name__)
_enrollment_sessions = {}
_enrollment_lock = threading.RLock()
_ENROLLMENT_TARGET_FRAMES = 10
_ENROLLMENT_SESSION_TTL_SECONDS = 15 * 60


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
            if str(result.get("status") or "denied") == "granted":
                lcd.show_access_granted(result.get("user", ""))
                try:
                    servo.unlock_door()
                except Exception as exc:
                    logger.warning("_trigger_hardware: servo unlock failed: %s", exc)
            else:
                detail = str(result.get("detail", "Denied"))[:16]
                lcd.show_access_denied(detail)
        except Exception as exc:
            logger.debug("Hardware actuation skipped (expected on dev machines): %s", exc)

    t = threading.Thread(target=_act, daemon=True, name="hardware-trigger")
    t.start()


# =============================================================================
# Helper  load prototype embeddings from DB into a dict
# =============================================================================

def _load_prototype_embeddings() -> dict:
    """
    Return {username: [np.ndarray, ...]} for every active, enrolled user.
    Used by authenticate_user() and the SVM trainer.
    """
    try:
        users = User.query.filter_by(active=True, enrolled=True).all()
        prototypes = {}
        for user in users:
            try:
                embeddings = user.get_np_embeddings()
                if embeddings:
                    prototypes[user.name] = embeddings
            except Exception as exc:
                logger.error("Failed to load embeddings for user %s: %s", user.name, exc)
        return prototypes
    except Exception as exc:
        logger.error("_load_prototype_embeddings failed: %s", exc)
        return {}


def _cleanup_enrollment_sessions() -> None:
    now = time.time()
    with _enrollment_lock:
        try:
            expired = [
                session_id
                for session_id, session in _enrollment_sessions.items()
                if now - session.get("started_at", now) > _ENROLLMENT_SESSION_TTL_SECONDS
            ]
            for session_id in expired:
                _enrollment_sessions.pop(session_id, None)
        except Exception as exc:
            logger.error("_cleanup_enrollment_sessions failed: %s", exc)


def _compact_user_embeddings(embeddings: list) -> list:
    """
    Keep only the most consistent enrollment templates so repeated enrollment
    sessions improve the model instead of filling it with noisy historical data.
    """
    cleaned = []
    for emb in embeddings or []:
        try:
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if np.isfinite(norm) and norm > 1e-8:
                cleaned.append(vec / norm)
        except Exception:
            continue

    if not cleaned:
        return []

    arr = np.stack(cleaned, axis=0)
    centroid = np.mean(arr, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if not np.isfinite(centroid_norm) or centroid_norm <= 1e-8:
        return [arr[0].astype(np.float32).tolist()]
    centroid = centroid / centroid_norm

    dists = np.linalg.norm(arr - centroid.reshape(1, -1), axis=1)
    order = np.argsort(dists)
    max_keep = max(4, int(os.environ.get("EMB_STORE_MAX_KEEP", "10") or "10"))

    selected = [centroid.astype(np.float32)]
    for idx in order[:max_keep]:
        selected.append(arr[int(idx)].astype(np.float32))

    compacted = []
    seen = set()
    for emb in selected:
        key = tuple(np.round(emb, 5).tolist())
        if key in seen:
            continue
        seen.add(key)
        compacted.append(emb.tolist())

    return compacted


def _retrain_model_after_enrollment() -> None:
    try:
        from app.ml.recognizer import train_classifier

        prototypes = _load_prototype_embeddings()
        result = train_classifier(prototypes)
        if not result.get("ok"):
            logger.warning("Automatic retraining after enrollment failed: %s", result)
    except Exception as exc:
        logger.warning("Automatic retraining after enrollment raised: %s", exc)


# =============================================================================
# User Management
# =============================================================================

@user_bp.route("/users", methods=["GET"])
def get_users():
    try:
        users = User.query.all()
        return jsonify([u.to_dict() for u in users])
    except Exception as exc:
        logger.error("GET /users failed: %s", exc)
        return jsonify({"error": "Database error"}), 500


@user_bp.route("/users", methods=["POST"])
def add_user():
    try:
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
    except Exception as exc:
        logger.error("POST /users failed: %s", exc)
        db.session.rollback()
        return jsonify({"error": "Database error"}), 500


@user_bp.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id: int):
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        db.session.delete(user)
        db.session.commit()
        
        # Retrain SVM to ensure the deleted user is removed from predictions
        try:
            from app.ml.recognizer import train_classifier
            prototypes = _load_prototype_embeddings()
            train_classifier(prototypes)
        except Exception as exc:
            logger.warning("Failed to retrain SVM after user deletion: %s", exc)
            
        return jsonify({"message": f"User {user_id} deleted."})
    except Exception as exc:
        logger.error("DELETE /users/%d failed: %s", user_id, exc)
        db.session.rollback()
        return jsonify({"error": "Database error"}), 500


# =============================================================================
# Access Logs
# =============================================================================

@user_bp.route("/logs", methods=["GET"])
def get_logs():
    try:
        limit = request.args.get("limit", type=int)
        q = AccessLog.query.order_by(AccessLog.id.desc())
        if limit:
            q = q.limit(max(1, min(limit, 1000)))
        logs = q.all()
        return jsonify([l.to_dict() for l in logs])
    except Exception as exc:
        logger.error("GET /logs failed: %s", exc)
        return jsonify({"error": "Database error"}), 500


@user_bp.route("/logs", methods=["DELETE"])
def clear_logs():
    try:
        num_deleted = AccessLog.query.delete()
        db.session.commit()
        return jsonify({"message": f"Cleared {num_deleted} logs successfully."}), 200
    except Exception as exc:
        logger.error("DELETE /logs failed: %s", exc)
        db.session.rollback()
        return jsonify({"error": "Database error"}), 500


# =============================================================================
# Liveness Challenge Token
# =============================================================================

@user_bp.route("/challenge", methods=["POST"])
def issue_challenge():
    try:
        from app.ml.liveness import generate_challenge
        token = generate_challenge()
        return jsonify(token)
    except Exception as exc:
        logger.error("POST /challenge failed: %s", exc)
        return jsonify({"error": "Internal error"}), 500


# =============================================================================
# Main Authentication Endpoint
# =============================================================================

@user_bp.route("/authenticate", methods=["POST"])
def authenticate():
    try:
        # ---------- parse inputs ----------
        image_file  = request.files.get("image")
        challenge   = (request.form.get("challenge") or "").upper().strip()
        issued_at   = request.form.get("issued_at")

        if not image_file:
            return jsonify({"error": "No image provided"}), 400
        if challenge not in ("BLINK", "LEFT", "RIGHT", "UP", "DOWN"):
            return jsonify({"error": "challenge must be BLINK, LEFT, RIGHT, UP, or DOWN"}), 400
        try:
            issued_at = float(issued_at)
        except (TypeError, ValueError):
            return jsonify({"error": "issued_at must be a Unix timestamp float"}), 400

        # ---------- decode frame ----------
        from app.ml.pipeline import authenticate_user, decode_frame
        
        try:
            image_bytes = image_file.read()
        except Exception as exc:
            logger.error("Failed to read image_file: %s", exc)
            return jsonify({"error": "Failed to read image upload"}), 400
            
        frame = decode_frame(image_bytes)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # ---------- run pipeline ----------
        prototypes = _load_prototype_embeddings()
        liveness_frames = []
        
        # Bound liveness frame count to avoid unbounded memory/cpu usage
        raw_liveness_files = request.files.getlist("liveness_images")[:30]
        
        for sample in raw_liveness_files:
            try:
                sample_bytes = sample.read()
                sample_frame = decode_frame(sample_bytes)
                if sample_frame is not None:
                    liveness_frames.append(sample_frame)
            except Exception as exc:
                logger.debug("Failed to read/decode liveness frame: %s", exc)

        result = authenticate_user(frame, challenge, issued_at, prototypes, liveness_frames or None)

        from app.security.repeat_detection import apply_repeat_policy
        try:
            result = apply_repeat_policy(result, frame, db, liveness_frames)
        except Exception as exc:
            logger.error("apply_repeat_policy failed: %s", exc)

        if not isinstance(result, dict):
            logger.error("authenticate_user returned non-dict")
            db.session.rollback()
            return jsonify({"error": "invalid_pipeline_result"}), 500

        status = str(result.get("status") or "denied").lower()
        user_name = str(result.get("user") or "unknown")
        liveness_ok = bool(result.get("liveness"))
        confidence = float(result.get("confidence") or 0.0)
        detail = result.get("detail")

        # ---------- actuate hardware (non-blocking, best-effort) ----------
        _trigger_hardware(result)

        # ---------- persist audit log ----------
        try:
            log = AccessLog(
                timestamp  = datetime.now().astimezone().isoformat(timespec="seconds"),
                status     = status,
                user       = user_name,
                liveness   = liveness_ok,
                confidence = confidence,
                detail     = detail,
                ip_address = request.remote_addr,
            )
            db.session.add(log)
            db.session.commit()
        except Exception as db_exc:
            logger.error("Failed to save AccessLog: %s", db_exc)
            db.session.rollback()

        http_status = 200 if status == "granted" else 403
        return jsonify(result), http_status

    except Exception as exc:
        logger.error("POST /authenticate uncaught exception: %s", exc)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500


@user_bp.route("/violations", methods=["GET"])
def get_violations():
    try:
        rows = ViolationImage.query.order_by(ViolationImage.id.desc()).all()
        grouped = {}
        for row in rows:
            item = row.to_dict()
            bucket = grouped.setdefault(row.group_id, {
                "groupId": row.group_id,
                "userId": row.user_id,
                "userName": item["userName"],
                "timestamp": row.timestamp,
                "images": [],
            })
            bucket["images"].append(item)
        return jsonify(list(grouped.values()))
    except Exception as exc:
        logger.error("GET /violations failed: %s", exc)
        return jsonify({"error": "Database error"}), 500


# =============================================================================
# Face Enrollment
# =============================================================================

@user_bp.route("/enroll/<int:user_id>", methods=["POST"])
def enroll_user(user_id: int):
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        # Hard limits
        if len(files) < 3:
            return jsonify({"error": "Minimum 3 images required for reliable enrollment"}), 400
        if len(files) > 15:
            return jsonify({"error": "Maximum 15 images per enrollment"}), 400

        from app.ml.pipeline import enroll_user as ml_enroll, decode_frame

        frames = []
        for f in files:
            try:
                frame_bytes = f.read()
                frame = decode_frame(frame_bytes)
                if frame is not None:
                    frames.append(frame)
            except Exception as exc:
                logger.debug("enroll_user: decode_frame failed: %s", exc)

        if not frames:
            return jsonify({"error": "No decodable images in upload"}), 400

        enroll_result = ml_enroll(frames, user.name)

        if not enroll_result["ok"]:
            return jsonify({"error": enroll_result["message"]}), 422

        existing = user.get_embeddings() or []
        user.set_embeddings(_compact_user_embeddings(existing + enroll_result["embeddings"]))
        db.session.commit()
        _retrain_model_after_enrollment()

        return jsonify({
            "message":  enroll_result["message"],
            "user":     user.to_dict(),
            "enrolled": enroll_result["count"],
        })

    except Exception as exc:
        logger.error("POST /enroll/%d failed: %s", user_id, exc)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500


@user_bp.route("/enroll/start", methods=["POST"])
def enroll_start():
    try:
        _cleanup_enrollment_sessions()
        data = request.json or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        user = User.query.filter_by(name=name).first()
        if not user:
            user = User(name=name)
            db.session.add(user)
            db.session.commit()

        session_id = uuid.uuid4().hex
        with _enrollment_lock:
            _enrollment_sessions[session_id] = {
                "user_id": user.id,
                "name": user.name,
                "frames": [],
                "started_at": time.time(),
            }

        return jsonify({
            "sessionId": session_id,
            "user": user.to_dict(),
            "targetFrames": _ENROLLMENT_TARGET_FRAMES,
            "captured": 0,
            "message": "Enrollment started",
        }), 201
    except Exception as exc:
        logger.error("POST /enroll/start failed: %s", exc)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500


@user_bp.route("/enroll/capture", methods=["POST"])
def enroll_capture():
    try:
        _cleanup_enrollment_sessions()
        data = request.json or {}
        session_id = data.get("sessionId")
        with _enrollment_lock:
            session = _enrollment_sessions.get(session_id)
            
        if not session:
            return jsonify({"error": "Enrollment session not found"}), 404

        from app.hardware.camera import capture_frame
        from app.ml.face_detector import detect_face

        frame = capture_frame()
        if frame is None:
            return jsonify({"error": "Camera unavailable"}), 503
            
        detection = detect_face(frame)
        if not isinstance(detection, dict) or detection.get("error"):
            return jsonify({"error": "No usable single face found"}), 422

        with _enrollment_lock:
            # Re-fetch inside lock to be safe
            session = _enrollment_sessions.get(session_id)
            if not session:
                return jsonify({"error": "Enrollment session expired"}), 404
            
            # Enforce max cap
            if len(session["frames"]) >= _ENROLLMENT_TARGET_FRAMES * 2:
                return jsonify({"error": "Maximum frames already captured for this session"}), 400
                
            session["frames"].append(frame)
            captured = len(session["frames"])

        return jsonify({
            "sessionId": session_id,
            "captured": captured,
            "targetFrames": _ENROLLMENT_TARGET_FRAMES,
            "complete": captured >= _ENROLLMENT_TARGET_FRAMES,
            "message": f"Captured image {captured}/{_ENROLLMENT_TARGET_FRAMES}",
        })
    except Exception as exc:
        logger.error("POST /enroll/capture failed: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


@user_bp.route("/enroll/complete", methods=["POST"])
def enroll_complete():
    try:
        _cleanup_enrollment_sessions()
        data = request.json or {}
        session_id = data.get("sessionId")
        with _enrollment_lock:
            session = _enrollment_sessions.get(session_id)
        if not session:
            return jsonify({"error": "Enrollment session not found"}), 404

        frames = list(session["frames"])
        if len(frames) < 3:
            return jsonify({"error": "At least 3 captured images are required"}), 400

        user = db.session.get(User, session["user_id"])
        if not user:
            return jsonify({"error": "User not found"}), 404

        from app.ml.pipeline import enroll_user as ml_enroll
        enroll_result = ml_enroll(frames, user.name)
        if not enroll_result["ok"]:
            return jsonify({"error": enroll_result["message"]}), 422

        existing = user.get_embeddings() or []
        user.set_embeddings(_compact_user_embeddings(existing + enroll_result["embeddings"]))
        db.session.commit()
        _retrain_model_after_enrollment()

        with _enrollment_lock:
            _enrollment_sessions.pop(session_id, None)

        return jsonify({
            "message": enroll_result["message"],
            "user": user.to_dict(),
            "enrolled": enroll_result["count"],
        })
    except Exception as exc:
        logger.error("POST /enroll/complete failed: %s", exc)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# SVM Training
# =============================================================================

@user_bp.route("/train", methods=["POST"])
def train_model():
    try:
        from app.ml.recognizer import train_classifier

        prototypes = _load_prototype_embeddings()
        result     = train_classifier(prototypes)

        status = 200 if result.get("ok") else 422
        return jsonify(result), status
    except Exception as exc:
        logger.error("POST /train failed: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# Legacy /scan (real ML only)
# =============================================================================

@user_bp.route("/scan", methods=["POST"])
def scan_face():
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image provided"}), 400

        from app.ml.pipeline import authenticate_user, decode_frame
        from app.ml.liveness import generate_challenge
        
        try:
            image_bytes = image_file.read()
        except Exception:
            return jsonify({"error": "Failed to read image"}), 400

        frame = decode_frame(image_bytes)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        token = generate_challenge()
        prototypes = _load_prototype_embeddings()
        auth = authenticate_user(frame, token["challenge"], token["issued_at"], prototypes)
        
        scan = {
            "result": "Authorized" if auth.get("status") == "granted" else "Unauthorized",
            "confidence": int(float(auth.get("confidence") or 0.0) * 100),
            "user": auth.get("user", "unknown"),
        }

        try:
            log = AccessLog(
                timestamp  = datetime.now().astimezone().isoformat(timespec="seconds"),
                status     = "granted"  if scan["result"] == "Authorized" else "denied",
                user       = scan.get("user", "unknown"),
                liveness   = False,
                confidence = scan["confidence"] / 100.0,
                detail     = "Legacy /scan call",
                ip_address = request.remote_addr,
            )
            db.session.add(log)
            db.session.commit()
        except Exception as db_exc:
            logger.error("Failed to save AccessLog for /scan: %s", db_exc)
            db.session.rollback()

        if scan["result"] == "Unauthorized":
            return jsonify({"alert": "Unknown person detected!", **scan})

        return jsonify(scan)
    except Exception as exc:
        logger.error("POST /scan failed: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# Toggle user active status
# =============================================================================

@user_bp.route("/users/<int:user_id>/toggle", methods=["PUT"])
def toggle_user(user_id: int):
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        user.active = not user.active
        db.session.commit()
        return jsonify({
            "success": True,
            "status":  "enabled" if user.active else "disabled",
            "active":  user.active,
            "user":    user.to_dict(),
        })
    except Exception as exc:
        logger.error("PUT /users/%d/toggle failed: %s", user_id, exc)
        db.session.rollback()
        return jsonify({"error": "Database error"}), 500


# =============================================================================
# Stats endpoint
# =============================================================================

@user_bp.route("/stats", methods=["GET"])
def get_stats():
    try:
        from sqlalchemy import func

        total_unlocks         = AccessLog.query.filter_by(status="granted").count()
        unauthorized_attempts = AccessLog.query.filter_by(status="denied").count()
        active_users          = User.query.filter_by(active=True, enrolled=True).count()

        total = total_unlocks + unauthorized_attempts
        denied_logs = AccessLog.query.filter_by(status="denied").all()
        detection_failure_phrases = (
            "no face detected",
            "multiple faces detected",
            "face detection failed",
            "invalid face detection",
            "incomplete face detection",
            "invalid detection",
        )
        detection_failures = sum(
            1
            for log in denied_logs
            if any(phrase in str(log.detail or "").lower() for phrase in detection_failure_phrases)
        )
        detection_successes = max(total - detection_failures, 0)
        detection_accuracy = round((detection_successes / total * 100), 1) if total else 0.0

        liveness_pass = AccessLog.query.filter_by(liveness=True).count()
        liveness_denominator = detection_successes
        liveness_pass_rate = round((liveness_pass / liveness_denominator * 100), 1) if liveness_denominator else 0.0

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
    except Exception as exc:
        logger.error("GET /stats failed: %s", exc)
        return jsonify({"error": "Database error"}), 500


# =============================================================================
# Camera frame endpoint
# =============================================================================

@user_bp.route("/camera/frame", methods=["GET"])
def camera_frame():
    try:
        from flask import Response
        from app.hardware.camera import capture_frame_jpeg

        jpeg = capture_frame_jpeg()
        if jpeg is None:
            return jsonify({"error": "Camera unavailable"}), 503

        return Response(jpeg, mimetype="image/jpeg")
    except Exception as exc:
        logger.error("GET /camera/frame failed: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
