# =============================================================================
# app/hardware/servo.py
# Servo Motor Controller  (Raspberry Pi GPIO / PWM)
#
# Drives a standard SG90/MG996R-style servo to unlock/lock a door.
#
# Wiring (example):
#   Servo signal pin  ->  GPIO 18  (BCM numbering, hardware PWM capable)
#   Servo VCC         ->  5 V rail
#   Servo GND         ->  GND
#
# Pulse widths:
#   Locked   position  -> 500  µs  (~0°)
#   Unlocked position  -> 2500 µs  (~180°)
#   (Tune LOCKED_DC / UNLOCKED_DC to match your servo and mechanical stop.)
# =============================================================================

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – adjust to suit your servo and GPIO setup
# ---------------------------------------------------------------------------
_SERVO_GPIO_PIN = 18          # BCM pin number (GPIO18 = hardware PWM0)
_PWM_FREQ_HZ    = 50          # Standard servo frequency

# Duty cycles for 50 Hz PWM:
#   duty = pulse_width_us / period_us * 100
#   period_us = 1_000_000 / 50 = 20_000 µs
_LOCKED_DC   = 2.5   # 500  µs  -> ~0°   (locked)
_UNLOCKED_DC = 12.5  # 2500 µs  -> ~180° (unlocked)

_UNLOCK_DURATION_S = 5.0      # seconds the door stays unlocked

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_pwm       = None
_gpio_ok   = False
_lock_timer: threading.Timer | None = None
_state_lock = threading.Lock()


def _init_gpio() -> bool:
    """Initialise RPi.GPIO and configure the servo pin. Returns True on success."""
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
    except (ImportError, RuntimeError) as exc:
        logger.warning("RPi.GPIO not available – servo disabled. (%s)", exc)
        return False


def _set_duty_cycle(dc: float) -> None:
    """Send a duty-cycle value to the servo (no-op if GPIO unavailable)."""
    if _gpio_ok and _pwm is not None:
        _pwm.ChangeDutyCycle(dc)


def _relock() -> None:
    """Timer callback – return the servo to the locked position."""
    # Acquire the state lock so this cannot race with a concurrent unlock_door() call.
    with _state_lock:
        logger.info("Auto-relocking door.")
        _set_duty_cycle(_LOCKED_DC)
        time.sleep(0.5)
        # Stop sending pulses to avoid servo jitter at rest
        if _gpio_ok and _pwm is not None:
            _pwm.ChangeDutyCycle(0)


def unlock_door(duration: float = _UNLOCK_DURATION_S) -> None:
    """
    Rotate the servo to the unlocked position and schedule an automatic
    relock after *duration* seconds.

    Safe to call from any thread; a previous timer is cancelled first.
    """
    global _lock_timer
    _init_gpio()

    with _state_lock:
        # Cancel any pending auto-relock
        if _lock_timer is not None and _lock_timer.is_alive():
            _lock_timer.cancel()

        logger.info("Unlocking door for %.1f s.", duration)
        _set_duty_cycle(_UNLOCKED_DC)

        _lock_timer = threading.Timer(duration, _relock)
        _lock_timer.daemon = True
        _lock_timer.start()


def lock_door() -> None:
    """
    Immediately rotate the servo to the locked position and cancel any
    pending auto-relock timer.
    """
    global _lock_timer
    _init_gpio()

    with _state_lock:
        if _lock_timer is not None and _lock_timer.is_alive():
            _lock_timer.cancel()
        logger.info("Locking door immediately.")
        _set_duty_cycle(_LOCKED_DC)
        time.sleep(0.5)
        if _gpio_ok and _pwm is not None:
            _pwm.ChangeDutyCycle(0)


def cleanup() -> None:
    """
    Release GPIO resources.  Call at application shutdown.
    """
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
