from __future__ import annotations

import re


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def canonical_model_id(model_id: str | None) -> str:
    if not isinstance(model_id, str):
        return ""
    return _NON_ALNUM.sub("", model_id.strip().lower())


def is_qwen35_a3_model(model_id: str | None) -> bool:
    canonical = canonical_model_id(model_id)
    return bool(canonical) and "qwen35" in canonical and "a3" in canonical

