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


def test_load_any_schema_resequences_fact_nums_contiguously() -> None:
    payload = {
        "images_dir": "data/pdf_images/test",
        "metadata": {},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"page_type": "other", "statement_type": None},
                "facts": [
                    {"bbox": [1, 2, 3, 4], "value": "12", "fact_num": 5, "path": []},
                    {"bbox": [5, 6, 7, 8], "value": "34", "fact_num": 5, "path": []},
                    {"bbox": [9, 10, 11, 12], "value": "56", "fact_num": 9, "path": []},
                ],
            }
        ],
    }
    out = load_any_schema(payload)
    assert [fact["fact_num"] for fact in out["pages"][0]["facts"]] == [1, 2, 3]


def test_load_any_schema_remaps_fact_equation_refs_when_fact_nums_shift() -> None:
    payload = {
        "images_dir": "data/pdf_images/test",
        "metadata": {},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"page_type": "other", "statement_type": None},
                "facts": [
                    {"bbox": [1, 2, 3, 4], "value": "900", "fact_num": 4, "path": []},
                    {"bbox": [5, 6, 7, 8], "value": "100", "fact_num": 1, "path": []},
                    {"bbox": [9, 10, 11, 12], "value": "20", "fact_num": 2, "path": []},
                    {
                        "bbox": [13, 14, 15, 16],
                        "value": "120",
                        "fact_num": 3,
                        "equation": "100 + 20",
                        "fact_equation": "f1 + f2",
                        "path": [],
                    },
                ],
            }
        ],
    }

    out = load_any_schema(payload)
    facts = out["pages"][0]["facts"]
    assert [fact["fact_num"] for fact in facts] == [1, 2, 3, 4]
    assert facts[3]["fact_equation"] == "f2 + f3"
