from __future__ import annotations

from typing import Any


def _is_negative_numeric_text(text: str) -> bool:
    text = str(text or "").strip()
    if not text or text == "-":
        return False
    return (text.startswith("(") and text.endswith(")")) or text.startswith("-")


def normalize_angle_bracketed_numeric_text(value: Any) -> str:
    raw = str(value or "").strip()
    if len(raw) < 3 or not raw.startswith("<") or not raw.endswith(">"):
        return raw

    inner = raw[1:-1].strip()
    if _is_negative_numeric_text(inner):
        return inner
    return raw


def derive_natural_sign_from_value_text(value: Any) -> str | None:
    text = normalize_angle_bracketed_numeric_text(value)
    if not text:
        return None
    if text == "-":
        return None
    if text.startswith("(") and text.endswith(")"):
        return "negative"
    if text.startswith("-"):
        return "negative"
    return "positive"
