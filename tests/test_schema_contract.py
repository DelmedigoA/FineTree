from __future__ import annotations

from pathlib import Path

from finetree_annotator.finetune.config import FinetuneConfig
from finetree_annotator.schema_contract import (
    CANONICAL_FACT_KEYS,
    EXTRACTED_FACT_KEYS,
    PAGE_META_KEYS,
    PROMPT_PAGE_META_KEYS,
    PROMPT_FACT_KEYS,
    REQUIRED_PROMPT_CANONICAL_KEYS,
    build_gemini_fill_updates_schema,
    default_gemini_autocomplete_prompt_template,
    default_gemini_fill_prompt_template,
    default_extraction_prompt_template,
)
from finetree_annotator.schemas import ExtractedFact, Fact, PageExtraction, PageMeta


def test_schema_contract_tracks_pydantic_models() -> None:
    assert PAGE_META_KEYS == tuple(PageMeta.model_fields.keys())
    assert CANONICAL_FACT_KEYS == tuple(Fact.model_fields.keys())
    assert EXTRACTED_FACT_KEYS == tuple(ExtractedFact.model_fields.keys())
    assert tuple(PageExtraction.model_fields.keys()) == ("schema_version", "images_dir", "metadata", "pages")


def test_schema_contract_required_prompt_keys_are_canonical() -> None:
    assert set(REQUIRED_PROMPT_CANONICAL_KEYS).issubset(set(CANONICAL_FACT_KEYS))
    assert "annotation_note" not in PROMPT_PAGE_META_KEYS
    assert "annotation_status" not in PROMPT_PAGE_META_KEYS
    assert "fact_num" in PROMPT_FACT_KEYS
    assert "equations" in PROMPT_FACT_KEYS
    assert "equation" not in PROMPT_FACT_KEYS
    assert "fact_equation" not in PROMPT_FACT_KEYS


def test_extraction_prompt_file_is_in_sync_with_schema_contract() -> None:
    prompt_path = Path("prompts/extraction_prompt.txt")
    assert prompt_path.read_text(encoding="utf-8").strip() == default_extraction_prompt_template()


def test_gemini_fill_prompt_file_is_in_sync_with_schema_contract() -> None:
    prompt_path = Path("prompts/gemini_fill_prompt.txt")
    assert prompt_path.read_text(encoding="utf-8").strip() == default_gemini_fill_prompt_template()


def test_gemini_autocomplete_prompt_file_is_in_sync_with_schema_contract() -> None:
    prompt_path = Path("prompts/gemini_autocomplete_prompt.txt")
    assert prompt_path.read_text(encoding="utf-8").strip() == default_gemini_autocomplete_prompt_template()


def test_finetune_config_default_fallback_template_comes_from_schema_contract() -> None:
    cfg = FinetuneConfig()
    assert cfg.prompt.fallback_template == default_extraction_prompt_template()


def test_prompt_contract_is_page_only() -> None:
    prompt = default_extraction_prompt_template()
    assert '"metadata"' not in prompt
    assert "company_name" not in prompt
    assert "report_year" not in prompt
    assert '"date"' not in prompt
    assert "Only return `pages` with page `image`, page `meta`, and page `facts`." in prompt
    assert "Only emit facts anchored on a visible numeric value" in prompt
    assert "נכסים שוטפים" in prompt
    assert 'If `comment_ref` seems unreasonably long, do not include the full text.' in prompt


def test_custom_schema_preview_is_page_only() -> None:
    from finetree_annotator.schema_contract import build_custom_extraction_prompt_template, build_custom_extraction_schema_preview

    preview = build_custom_extraction_schema_preview(
        page_meta_keys=("page_type", "title"),
        fact_keys=("value", "currency"),
    )
    prompt = build_custom_extraction_prompt_template(
        page_meta_keys=("page_type", "title"),
        fact_keys=("value", "currency"),
    )
    assert '"images_dir"' not in preview
    assert '"metadata"' not in preview
    assert '"pages"' not in preview
    assert '"image"' not in preview
    assert '"meta"' in preview
    assert '"facts"' in preview
    assert "page-level object" in prompt


def test_custom_no_bbox_prompt_omits_bbox_contract() -> None:
    from finetree_annotator.schema_contract import build_custom_extraction_prompt_template, build_custom_extraction_schema_preview

    preview = build_custom_extraction_schema_preview(
        page_meta_keys=("page_type", "title"),
        fact_keys=("value", "currency"),
        include_bbox=False,
    )
    prompt = build_custom_extraction_prompt_template(
        page_meta_keys=("page_type", "title"),
        fact_keys=("value", "currency"),
        include_bbox=False,
    )
    assert '"bbox"' not in preview
    assert "`bbox` must use original-image pixel coordinates" not in prompt
    assert "Selected fact keys:\n- value, currency" in prompt


def test_equation_schema_is_present_in_model_prompts() -> None:
    extraction_prompt = default_extraction_prompt_template()
    fill_prompt = build_gemini_fill_updates_schema(["value", "equations", "natural_sign", "row_role"])
    autocomplete_prompt = default_gemini_autocomplete_prompt_template()
    assert '"equations"' in extraction_prompt
    assert '"equation"' in extraction_prompt
    assert '"fact_equation"' in extraction_prompt
    assert "natural_sign" in extraction_prompt
    assert "row_role" in extraction_prompt
    assert "Do not emit legacy top-level `equation`, `fact_equation`, or `equation_children` keys inside facts." in extraction_prompt
    assert '"equations"' in fill_prompt
    assert '"equation"' in fill_prompt
    assert '"fact_equation"' in fill_prompt
    assert '"value"' in fill_prompt
    assert "natural_sign" in fill_prompt
    assert "row_role" in fill_prompt
    assert 'Never convert dash placeholders to `0`.' in default_gemini_fill_prompt_template()
    assert 'If `comment_ref` seems unreasonably long, do not include the full text.' in default_gemini_fill_prompt_template()
    assert "locked facts" in autocomplete_prompt.lower()
    assert "return only new missing facts" in autocomplete_prompt.lower()
    assert "original image pixel coordinates" in autocomplete_prompt.lower()
    assert "runtime rebuilds final contiguous numbering" in autocomplete_prompt.lower()
    assert 'If `comment_ref` seems unreasonably long, do not include the full text.' in autocomplete_prompt
    assert '"equations"' in autocomplete_prompt
    assert '"metadata"' not in autocomplete_prompt
    assert "balance_type" not in extraction_prompt
    assert "aggregation_role" not in extraction_prompt


def test_build_gemini_fill_updates_schema_only_contains_requested_fields() -> None:
    schema = build_gemini_fill_updates_schema(["value", "path"])
    assert '"value"' in schema
    assert '"path"' in schema
    assert '"natural_sign"' not in schema
    assert '"row_role"' not in schema
