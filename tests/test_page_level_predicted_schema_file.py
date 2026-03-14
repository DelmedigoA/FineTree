from __future__ import annotations

from pathlib import Path

from finetree_annotator.schema_contract import page_level_predicted_schema_document


def test_page_level_predicted_schema_file_is_synced() -> None:
    root = Path(__file__).resolve().parents[1]
    schema_path = root / "PAGE_LEVEL_PREDICTED_SCHEMA.md"
    assert schema_path.read_text(encoding="utf-8") == page_level_predicted_schema_document()
