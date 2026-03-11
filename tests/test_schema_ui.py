from __future__ import annotations

from finetree_annotator.schema_ui import enum_options, ui_descriptors


def test_schema_ui_enum_options_derive_from_registry() -> None:
    statement_opts = enum_options("page_meta", "statement_type")
    assert statement_opts[0] == ""
    assert "income_statement" in statement_opts
    assert "other_declaration" in statement_opts
    duration_opts = enum_options("fact", "duration_type")
    assert "recurrent" in duration_opts
    row_role_opts = enum_options("fact", "row_role")
    assert "detail" in row_role_opts
    assert "total" in row_role_opts
    assert enum_options("fact", "aggregation_role") == ("",)


def test_schema_ui_descriptors_include_expected_fields() -> None:
    descriptors = ui_descriptors()
    assert "page_meta.statement_type" in descriptors
    assert "metadata.entity_type" in descriptors
    assert "fact.row_role" in descriptors
    assert "fact.path_source" in descriptors
