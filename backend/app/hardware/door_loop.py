# =============================================================================
# app/hardware/door_loop.py
# Standalone Door-Lock Control Loop
#
# Runs independently of the Flask web server.
# Use this for the physical device operation on Raspberry Pi 5.
#
# Flow:
#   1. Show idle screen on LCD.
#   2. Wait for a face to appear in the camera frame.
#   3. Issue a liveness challenge, display it on LCD.
#   4. Capture a second frame after the user performs the head turn.
#   5. Run the full ML pipeline (detect → liveness → embed → recognise).
#   6. On GRANTED: show name on LCD, unlock servo for 5 s, log event.
#      On DENIED:  show reason on LCD, keep locked, log event.
#   7. Return to step 1.
#
# Launch:
#   python -m app.hardware.door_loop
# (Run inside the Flask app context so DB writes work.)
# =============================================================================

from __future__ import annotations

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def run_door_loop(app) -> None:
    """
    Blocking door-control loop.  Pass the Flask *app* instance so all DB
    operations execute inside the application context.

    Parameters
    ----------
    app : Flask
        The Flask application created by ``create_app()``.
    """
    from app.hardware import camera, lcd, servo
    from app.ml.pipeline     import authenticate_user
    from app.ml.liveness     import generate_challenge
    from app.models import db
    from app.models.log  import AccessLog
    from app.routes.user_routes import _load_prototype_embeddings

    logger.info("Door loop started.")
    lcd.show_idle()

    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5   # watchdog: restart idle after this many failures

    while True:
        try:
            _single_cycle(app, camera, lcd, servo, db, AccessLog,
                          authenticate_user, generate_challenge,
                          _load_prototype_embeddings)
            consecutive_errors = 0   # reset on any successful cycle
        except KeyboardInterrupt:
            logger.info("Door loop interrupted by user.")
            break
        except Exception as exc:
            consecutive_errors += 1
            logger.exception("Unexpected error in door loop (attempt %d/%d): %s",
                             consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc)
            lcd.display("  System error  ", "Restarting...   ")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                # Fail-safe: force the door locked and pause longer before retry
                logger.critical("Watchdog: %d consecutive errors. Forcing lock and pausing 30 s.",
                                consecutive_errors)
                try:
                    servo.lock_door()
                except Exception:
                    pass
                time.sleep(30)
                consecutive_errors = 0
            else:
                time.sleep(3)
            lcd.show_idle()


def _single_cycle(app, camera, lcd, servo, db, AccessLog,
                  authenticate_user, generate_challenge,
                  load_embeddings) -> None:
    """Execute one full authentication cycle."""

    # ------------------------------------------------------------------
    # Step 1: Wait for a face to appear
    # ------------------------------------------------------------------
    logger.debug("Waiting for face...")
    frame_with_face = camera.wait_for_face(timeout=30.0, poll_interval=0.15)

    if frame_with_face is None:
        # Nothing detected; show idle again and loop
        return

    # ------------------------------------------------------------------
    # Step 2: Issue liveness challenge & prompt user on LCD
    # ------------------------------------------------------------------
    token = generate_challenge()
    challenge    = token["challenge"]
    issued_at    = token["issued_at"]
    logger.info("Liveness challenge issued: %s", challenge)
    lcd.show_challenge(challenge)

    # Give the user time to perform the head turn before capture
    # (Pi ML inference adds ~8-12 s; the liveness timeout accounts for this)
    time.sleep(3.0)

    # ------------------------------------------------------------------
    # Step 3: Capture the response frame
    # ------------------------------------------------------------------
    lcd.show_processing()
    response_frame = camera.capture_frame()

    if response_frame is None:
        logger.warning("Camera capture failed during challenge.")
        lcd.display("Camera error    ", "Please retry    ")
        time.sleep(2)
        lcd.show_idle()
        return

    # ------------------------------------------------------------------
    # Step 4: Run the full ML pipeline
    # ------------------------------------------------------------------
    with app.app_context():
        prototypes = load_embeddings()
        result = authenticate_user(response_frame, challenge, issued_at, prototypes)

        status     = result["status"]
        user_name  = result["user"]
        liveness   = result["liveness"]
        confidence = result["confidence"]
        detail     = result.get("detail", "")

        logger.info(
            "Auth result: status=%s user=%s liveness=%s confidence=%.3f",
            status, user_name, liveness, confidence,
        )

        # ------------------------------------------------------------------
        # Step 5: Actuate hardware based on result
        # ------------------------------------------------------------------
        if status == "granted":
            lcd.show_access_granted(user_name)
            servo.unlock_door()          # auto-relocks after 5 s
        else:
            reason_map = {
                "no_face":        "No face found  ",
                "liveness_fail":  "Liveness fail   ",
                "embedding_fail": "Scan error      ",
            }
            lcd_reason = result.get("detail", "Unknown")[:16]
            lcd.show_access_denied(lcd_reason)

        # ------------------------------------------------------------------
        # Step 6: Persist audit log
        # ------------------------------------------------------------------
        log = AccessLog(
            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status     = status,
            user       = user_name,
            liveness   = liveness,
            confidence = confidence,
            detail     = detail,
            ip_address = "hardware",
        )
        db.session.add(log)
        db.session.commit()

    # Show result briefly before returning to idle
    time.sleep(3)
    lcd.show_idle()
