#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = REPO_ROOT / "finetree-v2_9.py"

DEFAULT_EXPORT_DIR = "artifacts/hf_finetree_3_0"
TRAIN_REPO_BASENAME = "FineTree-3.0-train"
VALIDATION_REPO_BASENAME = "FineTree-3.0-validation"


def _load_base_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finetree_v2_9_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_script()
_base.DEFAULT_EXPORT_DIR = DEFAULT_EXPORT_DIR
_base.TRAIN_REPO_BASENAME = TRAIN_REPO_BASENAME
_base.VALIDATION_REPO_BASENAME = VALIDATION_REPO_BASENAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push the public FineTree 3.0 split datasets.")
    parser.add_argument("--config", default=_base.DEFAULT_CONFIG_PATH)
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--token", default=None, help="HF token override.")
    parser.add_argument("--max-pixels", type=int, default=_base.DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--include_tabular_mixed_only",
        action="store_true",
        help="Exclude facts whose value_context is textual, keeping only tabular/mixed/unspecified facts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and export locally without pushing to HF.")
    return parser.parse_args(argv)


_base.parse_args = parse_args


def main(argv: list[str] | None = None) -> int:
    return int(_base.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
