from __future__ import annotations

from difflib import SequenceMatcher


def sequence_ratio(left: str, right: str) -> float:
    return float(SequenceMatcher(None, str(left), str(right)).ratio())
