#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.app import create_app
from app.models import db
from app.models.user import User  # noqa: F401
from app.models.log import AccessLog  # noqa: F401
from app.models.violation import ViolationImage  # noqa: F401
from sqlalchemy import text


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()
        db.session.execute(text("PRAGMA journal_mode=WAL"))
        db.session.execute(text("PRAGMA synchronous=NORMAL"))
        db.session.commit()
    print("Database initialized with WAL mode enabled.")


if __name__ == "__main__":
    main()
