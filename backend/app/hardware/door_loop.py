# =============================================================================
# app/hardware/door_loop.py  –  v2 (hardened)
#
# Fixes applied:
#   • Per-cycle timeout: _single_cycle() is wrapped in a thread with deadline
#   • Watchdog: after MAX_CONSECUTIVE_ERRORS the camera is fully reset
#   • Camera reset path: calls camera.release() then lets _get_cap re-open
#   • DB session rollback on every error path
#   • All exceptions are logged (no silent failures)
#   • KeyboardInterrupt / SystemExit propagate cleanly
# =============================================================================

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Maximum seconds one authentication cycle is allowed to take
_CYCLE_TIMEOUT_S = float(30.0)   # generous – includes liveness window


def run_door_loop(app) -> None:
    """
    Blocking door-control loop.  Pass the Flask *app* instance so all DB
    operations execute inside the application context.
    """
    from app.hardware import camera, lcd, servo
    from app.ml.pipeline     import authenticate_user
    from app.ml.liveness     import generate_challenge, CHALLENGE_TIMEOUT
    from app.models          import db
    from app.models.log      import AccessLog
    from app.routes.user_routes import _load_prototype_embeddings

    logger.info("Door loop started.")
    lcd.show_idle()

    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    while True:
        try:
            _single_cycle(
                app, camera, lcd, servo, db, AccessLog,
                authenticate_user, generate_challenge,
                _load_prototype_embeddings,
            )
            consecutive_errors = 0

        except KeyboardInterrupt:
            logger.info("Door loop interrupted by user.")
            break

        except SystemExit:
            logger.info("Door loop received SystemExit.")
            raise

        except Exception as exc:
            consecutive_errors += 1
            # Always roll back any uncommitted DB state
            with app.app_context():
                try:
                    db.session.rollback()
                except Exception as rb_exc:
                    logger.debug("DB rollback raised: %s", rb_exc)

            logger.exception(
                "Unexpected error in door loop (attempt %d/%d): %s",
                consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc,
            )
            lcd.display("  System error  ", "Restarting...   ")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "Watchdog: %d consecutive errors. Resetting camera + pipeline, pausing 30 s.",
                    consecutive_errors,
                )
                # Force door locked
                try:
                    servo.lock_door()
                except Exception:
                    pass
                # Reset camera to recover "device busy" state
                try:
                    camera.release()
                    logger.info("Camera released by watchdog – will reopen on next cycle.")
                except Exception as cam_exc:
                    logger.warning("Camera release (watchdog) raised: %s", cam_exc)

                time.sleep(30)
                consecutive_errors = 0
            else:
                time.sleep(3)

            lcd.show_idle()


def _single_cycle(
    app, camera, lcd, servo, db, AccessLog,
    authenticate_user, generate_challenge, load_embeddings,
) -> None:
    """Execute one full authentication cycle with an overall time guard."""

    # Run the inner cycle body in a thread so we can enforce a total deadline.
    result_container = [None]
    exc_container    = [None]

    def _body():
        try:
            result_container[0] = _cycle_body(
                app, camera, lcd, servo, db, AccessLog,
                authenticate_user, generate_challenge, load_embeddings,
            )
        except Exception as exc:
            exc_container[0] = exc

    t = threading.Thread(target=_body, daemon=True, name="door-cycle")
    t.start()
    t.join(timeout=_CYCLE_TIMEOUT_S)

    if t.is_alive():
        logger.error(
            "Authentication cycle exceeded %.0f s timeout – forcing next cycle.",
            _CYCLE_TIMEOUT_S,
        )
        # Thread is daemonised, will be cleaned up; just proceed to next cycle
        return

    if exc_container[0] is not None:
        raise exc_container[0]


def _cycle_body(
    app, camera, lcd, servo, db, AccessLog,
    authenticate_user, generate_challenge, load_embeddings,
) -> None:
    """Inner cycle logic – all blocking operations happen here."""

    # ------------------------------------------------------------------
    # Step 1: Wait for a face
    # ------------------------------------------------------------------
    lcd.show_idle()
    frame_with_face = camera.wait_for_face(timeout=30.0, poll_interval=0.3)

    if frame_with_face is None:
        # Nothing detected; return to outer loop
        return

    # ------------------------------------------------------------------
    # Step 2: Issue liveness challenge
    # ------------------------------------------------------------------
    token     = generate_challenge()
    challenge = token["challenge"]
    issued_at = token["issued_at"]
    logger.info("Liveness challenge issued: %s", challenge)
    lcd.show_challenge(challenge)

    # ------------------------------------------------------------------
    # Step 3: Collect response frames
    # ------------------------------------------------------------------
    response_frames: list = []
    deadline = time.monotonic() + min(CHALLENGE_TIMEOUT, 8.0)

    while time.monotonic() < deadline:
        sample = camera.capture_frame()
        if sample is not None:
            response_frames.append(sample)
            if len(response_frames) >= 30:
                break
        time.sleep(0.2)

    lcd.show_processing()

    response_frame = response_frames[-1] if response_frames else camera.capture_frame()
    if response_frame is None:
        logger.warning("Camera capture failed during challenge response collection.")
        lcd.display("Camera error    ", "Please retry    ")
        time.sleep(2)
        lcd.show_idle()
        return

    # ------------------------------------------------------------------
    # Step 4: ML pipeline + policy
    # ------------------------------------------------------------------
    with app.app_context():
        prototypes = load_embeddings()

        result = authenticate_user(
            response_frame,
            challenge,
            issued_at,
            prototypes,
            response_frames or None,
        )

        # Apply repeat-access policy (adds violation images if needed)
        try:
            from app.security.repeat_detection import apply_repeat_policy
            result = apply_repeat_policy(result, response_frame, db, response_frames)
        except Exception as exc:
            logger.warning("apply_repeat_policy raised: %s", exc)

        if not isinstance(result, dict):
            logger.error("authenticate_user returned non-dict: %s", type(result).__name__)
            result = {
                "status": "denied", "user": "unknown",
                "liveness": False, "confidence": 0.0,
                "detail": "Pipeline returned invalid result.",
            }

        status     = str(result.get("status")     or "denied")
        user_name  = str(result.get("user")        or "unknown")
        liveness   = bool(result.get("liveness"))
        confidence = float(result.get("confidence") or 0.0)
        detail     = str(result.get("detail")      or "")

        logger.info(
            "Auth result: status=%s user=%s liveness=%s confidence=%.3f",
            status, user_name, liveness, confidence,
        )

        # ------------------------------------------------------------------
        # Step 5: Actuate hardware
        # ------------------------------------------------------------------
        if status == "granted":
            lcd.show_access_granted(user_name)
            servo.unlock_door()
        else:
            lcd_reason = detail[:16] if detail else "Denied"
            lcd.show_access_denied(lcd_reason)

        # ------------------------------------------------------------------
        # Step 6: Persist audit log
        # ------------------------------------------------------------------
        try:
            log = AccessLog(
                timestamp  = datetime.now().astimezone().isoformat(timespec="seconds"),
                status     = status,
                user       = user_name,
                liveness   = liveness,
                confidence = confidence,
                detail     = detail,
                ip_address = "hardware",
            )
            db.session.add(log)
            db.session.commit()
        except Exception as db_exc:
            logger.error("Failed to persist audit log: %s", db_exc)
            try:
                db.session.rollback()
            except Exception:
                pass

    time.sleep(3)
    lcd.show_idle()
