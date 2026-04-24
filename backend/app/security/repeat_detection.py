from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2

from app.models.log import AccessLog
from app.models.user import User
from app.models.violation import ViolationImage
from app.models import db

REPEAT_WINDOW_MINUTES = int(os.environ.get("REPEAT_WINDOW_MINUTES", "10"))
REPEAT_THRESHOLD = int(os.environ.get("REPEAT_THRESHOLD", "5"))
VIOLATION_CAPTURE_COUNT = int(os.environ.get("VIOLATION_CAPTURE_COUNT", "3"))

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _BACKEND_ROOT / "data" / "violations"


def apply_repeat_policy(
    result: Dict[str, Any],
    frame,
    db,
    capture_frames: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """Track repetitive unauthorized attempts and persist evidence."""
    status = str(result.get("status") or "denied")
    user_name = str(result.get("user") or "unknown")

    # If granted, we do not log an intruder violation.
    if status == "granted":
        return result

    # Ensure an "unknown" user exists in DB to satisfy the Foreign Key constraint
    if user_name == "unknown":
        user = User.query.filter_by(name="unknown").first()
        if not user:
            user = User(name="unknown", active=False, enrolled=False)
            db.session.add(user)
            db.session.commit()
    else:
        user = User.query.filter_by(name=user_name).first()
        if not user:
            return result

    # Check for recent unauthorized attempts
    cutoff = datetime.now().astimezone() - timedelta(minutes=REPEAT_WINDOW_MINUTES)
    recent_count = (
        AccessLog.query
        .filter(AccessLog.user == user.name)
        .filter(AccessLog.status == "denied")
        .filter(AccessLog.timestamp >= cutoff.isoformat(timespec="seconds"))
        .count()
    )

    if recent_count + 1 < REPEAT_THRESHOLD:
        return result

    # Trigger Violation Image Capture
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    safe_stamp = timestamp.replace(":", "-")
    group_id = f"{user.id}_{safe_stamp}"
    evidence_paths = _save_evidence_images(user.id, safe_stamp, frame, capture_frames)

    for image_path in evidence_paths:
        db.session.add(ViolationImage(
            user_id=user.id,
            timestamp=timestamp,
            image_path=image_path,
            group_id=group_id,
        ))

    result = dict(result)
    result.update({
        "status": "denied",
        "repeatViolation": True,
        "violationImages": evidence_paths,
        "detail": (
            f"Intruder Alert: "
            f"{recent_count + 1} unauthorized attempts within {REPEAT_WINDOW_MINUTES} mins."
        ),
    })
    return result


def violation_root() -> Path:
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return _DATA_ROOT


def _save_evidence_images(user_id: int, safe_stamp: str, frame, capture_frames: Optional[Iterable]) -> list[str]:
    target_dir = violation_root() / str(user_id) / safe_stamp
    target_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if capture_frames:
        frames.extend([f for f in capture_frames if f is not None])
    if frame is not None:
        frames.append(frame)

    try:
        from app.hardware import camera
        while len(frames) < VIOLATION_CAPTURE_COUNT:
            captured = camera.capture_frame()
            if captured is not None:
                frames.append(captured)
            time.sleep(0.15)
    except Exception:
        pass

    if not frames and frame is not None:
        frames.append(frame)

    saved = []
    for index, image in enumerate(frames[:VIOLATION_CAPTURE_COUNT], start=1):
        rel_path = Path(str(user_id)) / safe_stamp / f"evidence_{index}.jpg"
        abs_path = violation_root() / rel_path
        if cv2.imwrite(str(abs_path), image):
            saved.append(rel_path.as_posix())
    return saved
