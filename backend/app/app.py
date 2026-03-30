import logging
from flask import Flask
from flask_cors import CORS
from app.models import db
from app.routes.user_routes import user_bp

# ---------------------------------------------------------------------------
# Package-level logging — propagates to gunicorn/werkzeug in production
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Database config (SQLite for development / Raspberry Pi)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Init DB extension
    db.init_app(app)

    # Register Blueprints (routes)
    app.register_blueprint(user_bp)

    # Ensure all tables exist (safe to call repeatedly — only creates missing ones)
    with app.app_context():
        db.create_all()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.route("/health")
    def health():
        return {
            "status":  "Backend running",
            "version": "2.0.0",
            "ml":      "face-recognition-pipeline",
        }

    return app
