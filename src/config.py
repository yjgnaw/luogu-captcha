"""Shared configuration constants for Luogu captcha recognition."""

from __future__ import annotations

# Character set used by generate.php (4-character captchas)
CHARSET: str = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnpqrstuvwxyz"
BLANK_INDEX: int = len(CHARSET)
NUM_CLASSES: int = BLANK_INDEX + 1  # charset + CTC blank

__all__ = ["CHARSET", "BLANK_INDEX", "NUM_CLASSES"]
