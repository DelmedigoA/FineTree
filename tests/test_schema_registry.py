from __future__ import annotations

from finetree_annotator.schema_registry import SchemaRegistry
from finetree_annotator.schemas import Document, Fact, Metadata, PageMeta


def test_schema_registry_model_specs_match_pydantic_fields() -> None:
    expected = {
        "metadata": tuple(Metadata.model_fields.keys()),
        "page_meta": tuple(PageMeta.model_fields.keys()),
        "fact": tuple(Fact.model_fields.keys()),
        "document": tuple(Document.model_fields.keys()),
    }
    for model_name, field_keys in expected.items():
        spec = SchemaRegistry.get_model_spec(model_name)
        assert spec.canonical_write_keys == field_keys


def test_schema_registry_prompt_contract_includes_schema_version_and_enums() -> None:
    extraction = SchemaRegistry.get_prompt_contract("extraction")
    patch = SchemaRegistry.get_prompt_contract("gemini_fill")
    assert extraction["schema_version"] == SchemaRegistry.current_version()
    assert "statement_types" in extraction["enums"]
    assert "other_declaration" in extraction["enums"]["statement_types"]
    assert extraction["enums"]["balance_types"] == ["debit", "credit"]
    assert extraction["enums"]["natural_signs"] == ["positive", "negative"]
    assert extraction["enums"]["row_roles"] == ["detail", "total"]
    assert extraction["enums"]["aggregation_roles"] == ["additive", "subtractive"]
    assert "equation" in patch["fact_patch_fields"]
    assert "balance_type" in patch["fact_patch_fields"]
    assert "natural_sign" in patch["fact_patch_fields"]
    assert "row_role" in patch["fact_patch_fields"]
    assert "aggregation_role" in patch["fact_patch_fields"]
    assert "value_type" in patch["fact_patch_fields"]
    assert "currency" in patch["fact_patch_fields"]
    assert "scale" in patch["fact_patch_fields"]
    assert "comment_ref" in patch["fact_patch_fields"]
    assert "schema_version" not in extraction["top_level_keys"]
    assert "annotation_note" not in extraction["page_meta_keys"]
    assert "annotation_status" not in extraction["page_meta_keys"]
    assert "fact_num" not in extraction["fact_keys"]
    assert "fact_equation" not in extraction["fact_keys"]


def test_schema_registry_prompt_contract_is_stable() -> None:
    first = SchemaRegistry.get_prompt_contract("extraction")
    second = SchemaRegistry.get_prompt_contract("extraction")
    assert first == second
