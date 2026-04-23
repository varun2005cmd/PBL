#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))


def check(label: str, fn) -> bool:
    try:
        fn()
        print(f"[OK] {label}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def main() -> int:
    from app.app import create_app
    from app.models import db
    from sqlalchemy import text

    app = create_app()
    ok = []
    ok.append(check("Flask app factory", lambda: app.test_client().get("/health")))
    with app.app_context():
        ok.append(check("SQLite query", lambda: db.session.execute(text("SELECT 1")).scalar()))
    ok.append(check("MediaPipe model file", lambda: (BACKEND / "app/ml/model_store/face_landmarker.task").stat()))
    ok.append(check("Face detector import", lambda: __import__("app.ml.face_detector")))
    ok.append(check("FaceNet import", lambda: __import__("app.ml.embedder")))
    ok.append(check("Camera module import", lambda: __import__("app.hardware.camera")))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
