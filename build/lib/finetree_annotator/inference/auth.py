from __future__ import annotations

import os
from typing import Optional


def resolve_hf_token_from_env(preferred_env: str = "FINETREE_HF_TOKEN") -> Optional[str]:
    candidates = (
        os.getenv(preferred_env),
        os.getenv("HF_TOKEN"),
        os.getenv("HUGGINGFACE_HUB_TOKEN"),
        os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    for token in candidates:
        if isinstance(token, str) and token.strip():
            return token.strip()
    return None

