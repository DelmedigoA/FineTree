from __future__ import annotations

from pathlib import Path

from finetree_annotator.finetune.config import FinetuneConfig
from finetree_annotator.schema_contract import (
    CANONICAL_FACT_KEYS,
    EXTRACTED_FACT_KEYS,
    PAGE_META_KEYS,
    REQUIRED_PROMPT_CANONICAL_KEYS,
    default_extraction_prompt_template,
)
from finetree_annotator.schemas import ExtractedFact, Fact, PageExtraction, PageMeta


def test_schema_contract_tracks_pydantic_models() -> None:
    assert PAGE_META_KEYS == tuple(PageMeta.model_fields.keys())
    assert CANONICAL_FACT_KEYS == tuple(Fact.model_fields.keys())
    assert EXTRACTED_FACT_KEYS == tuple(ExtractedFact.model_fields.keys())
    assert tuple(PageExtraction.model_fields.keys()) == ("images_dir", "metadata", "pages")


def test_schema_contract_required_prompt_keys_are_canonical() -> None:
    assert set(REQUIRED_PROMPT_CANONICAL_KEYS).issubset(set(CANONICAL_FACT_KEYS))


def test_extraction_prompt_file_is_in_sync_with_schema_contract() -> None:
    prompt_path = Path("prompts/extraction_prompt.txt")
    assert prompt_path.read_text(encoding="utf-8").strip() == default_extraction_prompt_template()


def test_finetune_config_default_fallback_template_comes_from_schema_contract() -> None:
    cfg = FinetuneConfig()
    assert cfg.prompt.fallback_template == default_extraction_prompt_template()


def test_metadata_fields_are_in_model_prompt() -> None:
    prompt = default_extraction_prompt_template()
    assert "company_name" in prompt
    assert "company_id" in prompt
    assert "report_year" in prompt
