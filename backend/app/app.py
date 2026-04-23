import logging
import atexit
import os
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from app.models import db
from app.routes.user_routes import user_bp

# ---------------------------------------------------------------------------
# Package-level logging  propagates to gunicorn/werkzeug in production
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log_dir = Path(__file__).resolve().parents[1] / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "backend.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_file_handler)


def _load_env_file() -> None:
    root = Path(__file__).resolve().parents[1].parent
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def create_app():
    _load_env_file()
    app = Flask(__name__)
    CORS(app)

    # Limit upload size to 8 MB to prevent unbounded memory usage / DoS
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

    # Database config (SQLite for development / Raspberry Pi)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # Enable WAL mode and a generous busy-timeout so the door-loop thread
    # and Flask request threads can both write without "database is locked" errors.
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {
            "check_same_thread": False,
            "timeout": 30,
        },
        "pool_pre_ping": True,
    }

    # Init DB extension
    db.init_app(app)


    # Register Blueprints (routes)
    app.register_blueprint(user_bp)
    from app.routes.hardware_routes import hardware_bp
    app.register_blueprint(hardware_bp)

    # Ensure all tables exist (safe to call repeatedly  only creates missing ones)
    with app.app_context():
        from app.models.violation import ViolationImage  # noqa: F401
        db.create_all()
        # Enable WAL journal mode for better concurrent read/write performance
        # (door-loop thread + Flask request threads writing simultaneously)
        from sqlalchemy import text as _text
        db.session.execute(_text("PRAGMA journal_mode=WAL"))
        db.session.execute(_text("PRAGMA synchronous=NORMAL"))
        db.session.commit()

    # ------------------------------------------------------------------
    # Hardware GPIO / I2C cleanup on process exit
    # ------------------------------------------------------------------
    def _hardware_cleanup():
        try:
            from app.hardware import servo, lcd, camera
            from app.ml.face_detector import release as release_detector
            servo.cleanup()
            lcd.cleanup()
            camera.release()
            release_detector()
        except Exception:
            pass

    atexit.register(_hardware_cleanup)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.route("/health")
    def health():
        try:
            from sqlalchemy import text as _text
            db.session.execute(_text("SELECT 1"))
            database = "connected"
        except Exception:
            database = "error"
        return jsonify({
            "status":  "Backend running",
            "version": "2.0.0",
            "ml":      "face-recognition-pipeline",
            "database": database,
        })

    @app.errorhandler(413)
    def request_entity_too_large(_exc):
        return jsonify({"error": "upload_too_large", "message": "Maximum upload size is 8 MB"}), 413

    @app.errorhandler(500)
    def internal_error(_exc):
        db.session.rollback()
        return jsonify({"error": "internal_server_error", "message": "Unexpected backend error"}), 500

    @app.route("/violations/image/<path:filename>")
    def violation_image(filename):
        from app.security.repeat_detection import violation_root
        return send_from_directory(violation_root(), filename)

    # ------------------------------------------------------------------
    # All hardware endpoints are now handled by hardware_bp blueprint
    # ------------------------------------------------------------------
    return app
