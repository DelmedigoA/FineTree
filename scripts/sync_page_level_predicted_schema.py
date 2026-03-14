#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from finetree_annotator.schema_contract import page_level_predicted_schema_document


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "PAGE_LEVEL_PREDICTED_SCHEMA.md"
    out_path.write_text(page_level_predicted_schema_document(), encoding="utf-8")
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
