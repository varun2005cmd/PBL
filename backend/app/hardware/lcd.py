# =============================================================================
# app/hardware/lcd.py
# I2C LCD Controller  (16×2 HD44780 with PCF8574 I2C backpack)
#
# Tested with common 1602 LCD + PCF8574 I2C backpack modules sold for Pi.
#
# Wiring:
#   LCD SDA -> GPIO 2  (SDA1, physical pin 3)
#   LCD SCL -> GPIO 3  (SCL1, physical pin 5)
#   LCD VCC -> 5 V
#   LCD GND -> GND
#
# Enable I2C on the Pi:
#   sudo raspi-config -> Interface Options -> I2C -> Enable
#
# Find the I2C address:
#   sudo i2cdetect -y 1   (usually 0x27 or 0x3F)
# =============================================================================

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_I2C_ADDR  = 0x27      # change to 0x3F if i2cdetect shows that address
_I2C_BUS   = 1         # /dev/i2c-1  (standard on Pi 2+)
_LCD_COLS  = 16
_LCD_ROWS  = 2

# HD44780 command constants
_LCD_CHR   = 1         # character mode
_LCD_CMD   = 0         # command mode
_LCD_BACKLIGHT = 0x08
_ENABLE    = 0b00000100

_LCD_LINE_1 = 0x80
_LCD_LINE_2 = 0xC0

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_bus    = None
_lcd_ok = False
_write_lock = threading.Lock()

# Default lines shown when the system is idle
_IDLE_LINE1 = "  Door Lock Sys "
_IDLE_LINE2 = " Scan your face "


def _init_lcd() -> bool:
    """Initialise smbus2 and the LCD controller. Returns True on success."""
    global _bus, _lcd_ok
    if _lcd_ok:
        return True
    try:
        import smbus2
        _bus = smbus2.SMBus(_I2C_BUS)
        # Initialise HD44780 in 4-bit mode
        _lcd_byte(_LCD_CMD, 0x33)
        _lcd_byte(_LCD_CMD, 0x32)
        _lcd_byte(_LCD_CMD, 0x06)
        _lcd_byte(_LCD_CMD, 0x0C)
        _lcd_byte(_LCD_CMD, 0x28)
        _lcd_byte(_LCD_CMD, 0x01)
        time.sleep(0.05)
        _lcd_ok = True
        logger.info("LCD initialised at I2C addr 0x%02X.", _I2C_ADDR)
        return True
    except (ImportError, OSError) as exc:
        logger.warning("LCD not available – display disabled. (%s)", exc)
        return False


def _lcd_byte(mode: int, bits: int) -> None:
    """Send one byte to the LCD (4-bit nibble mode)."""
    if not _lcd_ok and mode != _LCD_CMD:
        return
    if _bus is None:
        return
    bits_high = mode | (bits & 0xF0)        | _LCD_BACKLIGHT
    bits_low  = mode | ((bits << 4) & 0xF0) | _LCD_BACKLIGHT
    _bus.write_byte(_I2C_ADDR, bits_high)
    _lcd_toggle_enable(bits_high)
    _bus.write_byte(_I2C_ADDR, bits_low)
    _lcd_toggle_enable(bits_low)


def _lcd_toggle_enable(bits: int) -> None:
    time.sleep(0.0005)
    _bus.write_byte(_I2C_ADDR, bits | _ENABLE)
    time.sleep(0.0005)
    _bus.write_byte(_I2C_ADDR, bits & ~_ENABLE)
    time.sleep(0.0005)


def _write_line(text: str, line: int) -> None:
    """Write a single line of text (auto-padded to 16 chars)."""
    addr = _LCD_LINE_1 if line == 1 else _LCD_LINE_2
    _lcd_byte(_LCD_CMD, addr)
    text = text[:_LCD_COLS].ljust(_LCD_COLS)
    for char in text:
        _lcd_byte(_LCD_CHR, ord(char))


def display(line1: str = "", line2: str = "") -> None:
    """
    Write two lines to the LCD.  Safe to call from any thread.
    No-op if the LCD hardware is not available.
    """
    if not _init_lcd():
        logger.debug("LCD display() skipped – hardware unavailable.")
        return
    with _write_lock:
        _write_line(line1, 1)
        _write_line(line2, 2)


def show_idle() -> None:
    """Display the default idle / standby message."""
    display(_IDLE_LINE1, _IDLE_LINE2)


def show_access_granted(name: str) -> None:
    """Display an 'access granted' message with the recognised user name."""
    short = name[:16] if name else "User"
    display("Access Granted  ", short)


def show_access_denied(reason: str = "Denied") -> None:
    """Display an 'access denied' message."""
    display("Access Denied   ", reason[:16])


def show_challenge(direction: str) -> None:
    """Prompt the user to perform the liveness head-turn challenge."""
    display("Liveness check: ", f"Turn {direction}   ")


def show_processing() -> None:
    """Display a processing indicator while the pipeline runs."""
    display("Processing...   ", "Please wait     ")


def clear() -> None:
    """Clear the LCD display."""
    if _lcd_ok:
        with _write_lock:
            _lcd_byte(_LCD_CMD, 0x01)
            time.sleep(0.05)


def cleanup() -> None:
    """Close the I2C bus. Call at application shutdown."""
    global _bus, _lcd_ok
    if _lcd_ok and _bus is not None:
        try:
            clear()
            _bus.close()
            _lcd_ok = False
            logger.info("LCD I2C bus closed.")
        except Exception as exc:
            logger.warning("LCD cleanup error: %s", exc)
