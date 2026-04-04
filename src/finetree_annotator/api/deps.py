"""Shared dependencies and configuration for the web API."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..workspace import DEFAULT_DATA_ROOT

_data_root: Path = DEFAULT_DATA_ROOT


def get_data_root() -> Path:
    return _data_root


def set_data_root(root: Path) -> None:
    global _data_root
    _data_root = Path(root)
