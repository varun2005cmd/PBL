import logging
import atexit
from flask import Flask, jsonify
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


def create_app():
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
        return jsonify({
            "status":  "Backend running",
            "version": "2.0.0",
            "ml":      "face-recognition-pipeline",
        })

    # ------------------------------------------------------------------
    # All hardware endpoints are now handled by hardware_bp blueprint
    # ------------------------------------------------------------------
    return app
