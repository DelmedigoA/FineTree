from __future__ import annotations

from finetree_annotator.schema_io import load_any_schema, payload_requires_migration, save_canonical
from finetree_annotator.schema_registry import CURRENT_SCHEMA_VERSION


def test_load_any_schema_normalizes_legacy_payload_to_canonical() -> None:
    legacy_payload = {
        "document_meta": {"language": "HE"},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "notes", "entity_name": "Acme"},
                "facts": [
                    {
                        "bbox": [1, 2, 3, 4],
                        "value": "10",
                        "comment": "legacy comment",
                        "note_reference": "N1",
                        "is_beur": True,
                        "beur_num": "5",
                        "path": [],
                    }
                ],
            }
        ],
    }
    out = load_any_schema(legacy_payload)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["metadata"]["language"] == "he"
    page = out["pages"][0]
    assert page["meta"]["page_type"] == "statements"
    assert page["meta"]["statement_type"] == "notes_to_financial_statements"
    fact = page["facts"][0]
    assert fact["comment_ref"] == "legacy comment"
    assert fact["note_ref"] == "N1"
    assert fact["note_num"] == 5
    assert "comment" not in fact
    assert "note_reference" not in fact
    assert "beur_num" not in fact


def test_save_canonical_emits_schema_version_and_canonical_keys_only() -> None:
    payload = {
        "images_dir": "data/pdf_images/test",
        "metadata": {},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"page_type": "other", "statement_type": None},
                "facts": [{"bbox": [1, 2, 3, 4], "value": "12", "path": [], "ref_note": "X"}],
            }
        ],
    }
    out = save_canonical(payload)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    fact = out["pages"][0]["facts"][0]
    assert fact["note_ref"] == "X"
    assert "ref_note" not in fact


def test_payload_requires_migration_for_missing_schema_version() -> None:
    payload = {"images_dir": "x", "metadata": {}, "pages": []}
    assert payload_requires_migration(payload) is True
