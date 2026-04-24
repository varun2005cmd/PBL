# =============================================================================
# app/hardware/servo.py  –  v2 (hardened)
# =============================================================================

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SERVO_GPIO_PIN = 18          # BCM pin number (GPIO18 = hardware PWM0)
_PWM_FREQ_HZ    = 50          # Standard servo frequency

_MIN_DC = float(os.environ.get("SERVO_MIN_DUTY", "2.5"))
_MAX_DC = float(os.environ.get("SERVO_MAX_DUTY", "12.5"))
_LOCKED_ANGLE = float(os.environ.get("SERVO_LOCK_ANGLE", "0"))
_UNLOCKED_ANGLE = float(os.environ.get("SERVO_UNLOCK_ANGLE", "90"))

_LOCKED_DC = _MIN_DC + ((_MAX_DC - _MIN_DC) * (max(0.0, min(180.0, _LOCKED_ANGLE)) / 180.0))
_UNLOCKED_DC = _MIN_DC + ((_MAX_DC - _MIN_DC) * (max(0.0, min(180.0, _UNLOCKED_ANGLE)) / 180.0))

_UNLOCK_DURATION_S = 5.0

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_pwm       = None
_gpio_ok   = False
_lock_timer: threading.Timer | None = None
_state_lock = threading.RLock()
_door_is_unlocked = False


def _init_gpio() -> bool:
    global _pwm, _gpio_ok
    if _gpio_ok:
        return True
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(_SERVO_GPIO_PIN, GPIO.OUT)
        _pwm = GPIO.PWM(_SERVO_GPIO_PIN, _PWM_FREQ_HZ)
        _pwm.start(_LOCKED_DC)
        _gpio_ok = True
        logger.info("Servo GPIO initialised on pin %d.", _SERVO_GPIO_PIN)
        return True
    except Exception as exc:
        logger.warning("RPi.GPIO not available – servo disabled. (%s)", exc)
        return False


def _set_duty_cycle(dc: float) -> None:
    if _gpio_ok and _pwm is not None:
        try:
            _pwm.ChangeDutyCycle(dc)
        except Exception as exc:
            global _gpio_ok
            _gpio_ok = False
            logger.warning("Servo write failed, disabling servo: %s", exc)


def _relock() -> None:
    global _door_is_unlocked
    with _state_lock:
        logger.info("Auto-relocking door.")
        _set_duty_cycle(_LOCKED_DC)
        time.sleep(0.5)
        if _gpio_ok and _pwm is not None:
            try:
                _pwm.ChangeDutyCycle(0)
            except Exception:
                pass
        _door_is_unlocked = False


def unlock_door(duration: float = _UNLOCK_DURATION_S) -> None:
    global _lock_timer, _door_is_unlocked
    _init_gpio()

    with _state_lock:
        if _lock_timer is not None and _lock_timer.is_alive():
            _lock_timer.cancel()

        duration = max(1.0, min(float(duration), 30.0))
        logger.info("Unlocking door for %.1f s.", duration)
        _set_duty_cycle(_UNLOCKED_DC)
        _door_is_unlocked = True

        _lock_timer = threading.Timer(duration, _relock)
        _lock_timer.daemon = True
        _lock_timer.start()


def lock_door() -> None:
    global _lock_timer, _door_is_unlocked
    _init_gpio()

    with _state_lock:
        if _lock_timer is not None and _lock_timer.is_alive():
            _lock_timer.cancel()
        logger.info("Locking door immediately.")
        _set_duty_cycle(_LOCKED_DC)
        time.sleep(0.5)
        if _gpio_ok and _pwm is not None:
            try:
                _pwm.ChangeDutyCycle(0)
            except Exception:
                pass
        _door_is_unlocked = False


def cleanup() -> None:
    global _gpio_ok
    if _gpio_ok:
        try:
            import RPi.GPIO as GPIO
            if _pwm is not None:
                _pwm.stop()
            GPIO.cleanup()
            _gpio_ok = False
            logger.info("Servo GPIO cleaned up.")
        except Exception as exc:
            logger.warning("GPIO cleanup error: %s", exc)


def status() -> dict:
    return {
        "available": _gpio_ok,
        "unlocked": _door_is_unlocked,
        "pin": _SERVO_GPIO_PIN,
        "pwm_hz": _PWM_FREQ_HZ,
    }
