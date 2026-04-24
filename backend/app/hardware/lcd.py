# =============================================================================
# app/hardware/lcd.py  –  v2 (hardened)
# =============================================================================

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_I2C_ADDR_ENV = os.environ.get("LCD_I2C_ADDR", "")
_I2C_ADDR  = int(_I2C_ADDR_ENV, 16) if _I2C_ADDR_ENV else None
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
_active_addr: Optional[int] = None
_write_lock = threading.Lock()

_IDLE_LINE1 = "  Door Lock Sys "
_IDLE_LINE2 = " Scan your face "


def _init_lcd() -> bool:
    global _bus, _lcd_ok, _active_addr
    if _lcd_ok:
        return True
    try:
        import smbus2
        _bus = smbus2.SMBus(_I2C_BUS)

        candidates = [_I2C_ADDR] if _I2C_ADDR is not None else []
        for addr in (0x27, 0x3F):
            if addr not in candidates:
                candidates.append(addr)

        for addr in candidates:
            if addr is None:
                continue
            try:
                _active_addr = addr
                _lcd_byte(_LCD_CMD, 0x33)
                _lcd_byte(_LCD_CMD, 0x32)
                _lcd_byte(_LCD_CMD, 0x06)
                _lcd_byte(_LCD_CMD, 0x0C)
                _lcd_byte(_LCD_CMD, 0x28)
                _lcd_byte(_LCD_CMD, 0x01)
                time.sleep(0.05)
                _lcd_ok = True
                logger.info("LCD initialised at I2C addr 0x%02X.", _active_addr)
                return True
            except OSError:
                _lcd_ok = False

        logger.warning("LCD not detected at I2C addresses 0x27 or 0x3F.")
        return False
    except Exception as exc:
        logger.warning("LCD not available – display disabled. (%s)", exc)
        return False


def _lcd_byte(mode: int, bits: int) -> None:
    if not _lcd_ok and mode != _LCD_CMD:
        return
    if _bus is None:
        return
    addr = _active_addr if _active_addr is not None else _I2C_ADDR
    if addr is None:
        return
    try:
        bits_high = mode | (bits & 0xF0)        | _LCD_BACKLIGHT
        bits_low  = mode | ((bits << 4) & 0xF0) | _LCD_BACKLIGHT
        _bus.write_byte(addr, bits_high)
        _lcd_toggle_enable(bits_high)
        _bus.write_byte(addr, bits_low)
        _lcd_toggle_enable(bits_low)
    except OSError as exc:
        global _lcd_ok
        _lcd_ok = False
        logger.warning("LCD write failed, disabling display: %s", exc)


def _lcd_toggle_enable(bits: int) -> None:
    addr = _active_addr if _active_addr is not None else _I2C_ADDR
    if _bus is None or addr is None:
        return
    try:
        time.sleep(0.0005)
        _bus.write_byte(addr, bits | _ENABLE)
        time.sleep(0.0005)
        _bus.write_byte(addr, bits & ~_ENABLE)
        time.sleep(0.0005)
    except OSError:
        pass


def _write_line(text: str, line: int) -> None:
    addr = _LCD_LINE_1 if line == 1 else _LCD_LINE_2
    _lcd_byte(_LCD_CMD, addr)
    text = text[:_LCD_COLS].ljust(_LCD_COLS)
    for char in text:
        _lcd_byte(_LCD_CHR, ord(char))


def display(line1: str = "", line2: str = "") -> None:
    if not _init_lcd():
        return
    with _write_lock:
        _write_line(line1, 1)
        _write_line(line2, 2)


def show_idle() -> None:
    display(_IDLE_LINE1, _IDLE_LINE2)

def show_access_granted(name: str) -> None:
    short = name[:16] if name else "User"
    display("Access Granted  ", short)

def show_access_denied(reason: str = "Denied") -> None:
    display("Access Denied   ", reason[:16])

def show_challenge(direction: str) -> None:
    prompts = {
        "BLINK": "Blink twice",
        "LEFT": "Look left",
        "RIGHT": "Look right",
        "UP": "Look up",
        "DOWN": "Look down",
    }
    display("Liveness check: ", prompts.get(direction, str(direction))[:16])

def show_processing() -> None:
    display("Processing...   ", "Please wait     ")

def clear() -> None:
    if _lcd_ok:
        with _write_lock:
            _lcd_byte(_LCD_CMD, 0x01)
            time.sleep(0.05)


def cleanup() -> None:
    global _bus, _lcd_ok
    if _lcd_ok and _bus is not None:
        try:
            clear()
            _bus.close()
            _lcd_ok = False
            logger.info("LCD I2C bus closed.")
        except Exception as exc:
            logger.warning("LCD cleanup error: %s", exc)


def status() -> dict:
    return {
        "available": _lcd_ok,
        "address": f"0x{_active_addr:02X}" if _active_addr is not None else None,
        "bus": _I2C_BUS,
    }
