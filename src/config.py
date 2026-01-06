"""Shared configuration constants for Luogu captcha recognition."""

from __future__ import annotations

# Character set used by generate.php (4-character captchas)
CHARSET: str = "abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX: int = len(CHARSET)
NUM_CLASSES: int = BLANK_INDEX + 1  # charset + CTC blank

__all__ = ["CHARSET", "BLANK_INDEX", "NUM_CLASSES"]
