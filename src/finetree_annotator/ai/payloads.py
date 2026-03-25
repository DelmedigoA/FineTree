from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from ..model_prompt_serialization import MODEL_PROMPT_MODE, build_single_page_payload, serialize_schema_mode_payload
from ..schema_contract import (
    build_gemini_fill_updates_schema,
    default_extraction_prompt_template,
    default_gemini_autocomplete_prompt_template,
    default_gemini_fill_prompt_template,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROMPTS_ROOT = REPO_ROOT / "prompts"
DEFAULT_EXTRACTION_PROMPT_PATH = PROMPTS_ROOT / "extraction_prompt.txt"
DEFAULT_GEMINI_FILL_PROMPT_PATH = PROMPTS_ROOT / "gemini_fill_prompt.txt"
DEFAULT_GEMINI_AUTOCOMPLETE_PROMPT_PATH = PROMPTS_ROOT / "gemini_autocomplete_prompt.txt"
LEGACY_EXTRACTION_PROMPT_PATH = REPO_ROOT / "prompt.txt"


def _first_existing_path(candidates: list[Path]) -> Optional[Path]:
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def resolve_extraction_prompt_path() -> Optional[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("FINETREE_PROMPT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path.cwd() / "prompts" / "extraction_prompt.txt",
            DEFAULT_EXTRACTION_PROMPT_PATH,
            Path.cwd() / "prompt.txt",
            LEGACY_EXTRACTION_PROMPT_PATH,
        ]
    )
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "prompts" / "extraction_prompt.txt")
        candidates.append(parent / "prompt.txt")
    return _first_existing_path(candidates)


def resolve_gemini_fill_prompt_path() -> Optional[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("FINETREE_GEMINI_FILL_PROMPT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([Path.cwd() / "prompts" / "gemini_fill_prompt.txt", DEFAULT_GEMINI_FILL_PROMPT_PATH])
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "prompts" / "gemini_fill_prompt.txt")
    return _first_existing_path(candidates)


def resolve_gemini_autocomplete_prompt_path() -> Optional[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("FINETREE_GEMINI_AUTOCOMPLETE_PROMPT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([Path.cwd() / "prompts" / "gemini_autocomplete_prompt.txt", DEFAULT_GEMINI_AUTOCOMPLETE_PROMPT_PATH])
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "prompts" / "gemini_autocomplete_prompt.txt")
    return _first_existing_path(candidates)


def load_prompt_template(path: Optional[Path], *, fallback_text: str) -> str:
    if path is None:
        return fallback_text
    return path.read_text(encoding="utf-8")


def extraction_prompt_template() -> str:
    return load_prompt_template(resolve_extraction_prompt_path(), fallback_text=default_extraction_prompt_template())


def gemini_fill_prompt_template() -> str:
    return load_prompt_template(resolve_gemini_fill_prompt_path(), fallback_text=default_gemini_fill_prompt_template())


def gemini_autocomplete_prompt_template() -> str:
    return load_prompt_template(
        resolve_gemini_autocomplete_prompt_path(),
        fallback_text=default_gemini_autocomplete_prompt_template(),
    )


def build_extraction_prompt(
    template: str,
    page_image_path: Path,
    *,
    image_dimensions: Optional[tuple[float, float]],
) -> str:
    prompt = template.replace("{{PAGE_IMAGE}}", str(page_image_path))
    prompt = prompt.replace("{{IMAGE_NAME}}", page_image_path.name)
    image_size = (
        f"{int(image_dimensions[0])} x {int(image_dimensions[1])} pixels"
        if image_dimensions is not None
        else "unknown"
    )
    return prompt.replace("{{IMAGE_DIMENSIONS}}", image_size)


def build_page_prompt_payload(
    *,
    page_name: str,
    page_meta: dict[str, Any],
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    return build_single_page_payload(
        page_name=page_name,
        page_meta=page_meta,
        facts=facts,
        mode=MODEL_PROMPT_MODE,
    )


def build_gemini_fill_request_payload(
    *,
    page_name: str,
    page_meta: dict[str, Any],
    ordered_fact_payloads: list[dict[str, Any]],
    selected_fact_nums: list[int],
    selected_fact_fields: set[str],
    include_statement_type: bool,
) -> dict[str, Any]:
    payload_meta = dict(page_meta or {})
    if include_statement_type:
        payload_meta["statement_type"] = None

    facts_out: list[dict[str, Any]] = []
    selected_lookup = set(selected_fact_nums)
    for payload in ordered_fact_payloads:
        fact_num = payload.get("fact_num")
        if fact_num not in selected_lookup:
            continue
        redacted_fact = dict(payload)
        redacted_fact.pop("bbox", None)
        redacted_fact.pop("fact_num", None)
        for field_name in selected_fact_fields:
            if field_name in redacted_fact:
                redacted_fact[field_name] = None
        facts_out.append(
            {
                "fact_num": fact_num,
                "bbox": payload.get("bbox"),
                **redacted_fact,
            }
        )

    return build_page_prompt_payload(page_name=page_name, page_meta=payload_meta, facts=facts_out)


def build_gemini_fill_prompt(
    template: str,
    *,
    request_payload: dict[str, Any],
    selected_fact_fields: set[str],
    include_statement_type: bool,
) -> str:
    fact_fields = ", ".join(sorted(selected_fact_fields)) if selected_fact_fields else "none"
    meta_fields = "statement_type" if include_statement_type else "none"
    if include_statement_type:
        meta_updates_schema = (
            '{\n'
            '    "statement_type": '
            '"balance_sheet|income_statement|cash_flow_statement|statement_of_changes_in_equity|'
            'notes_to_financial_statements|board_of_directors_report|auditors_report|'
            'statement_of_activities|other_declaration|null"\n'
            "  }"
        )
    else:
        meta_updates_schema = "{}"
    fact_updates_schema = build_gemini_fill_updates_schema(sorted(selected_fact_fields))
    request_json = serialize_schema_mode_payload(request_payload)
    prompt = template.replace("{{FACT_FIELDS}}", fact_fields)
    prompt = prompt.replace("{{META_FIELDS}}", meta_fields)
    prompt = prompt.replace("{{META_UPDATES_SCHEMA}}", meta_updates_schema)
    prompt = prompt.replace("{{FACT_UPDATES_SCHEMA}}", fact_updates_schema)
    return prompt.replace("{{REQUEST_JSON}}", request_json)


def build_gemini_autocomplete_request_payload(
    *,
    page_name: str,
    page_meta: dict[str, Any],
    ordered_fact_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    facts_out: list[dict[str, Any]] = []
    for payload in ordered_fact_payloads:
        facts_out.append(
            {
                "fact_num": payload.get("fact_num"),
                "bbox": payload.get("bbox"),
                **{key: value for key, value in payload.items() if key not in {"bbox", "fact_num"}},
            }
        )
    return build_page_prompt_payload(page_name=page_name, page_meta=page_meta, facts=facts_out)


def build_gemini_autocomplete_prompt(
    template: str,
    *,
    request_payload: dict[str, Any],
    image_dimensions: Optional[tuple[float, float]],
) -> str:
    request_json = serialize_schema_mode_payload(request_payload)
    first_page: dict[str, Any] = {}
    pages = request_payload.get("pages")
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        first_page = {
            "image": pages[0].get("image"),
            "meta": pages[0].get("meta"),
            "facts": pages[0].get("facts"),
        }
    seed_page_json = json.dumps(first_page, ensure_ascii=False, indent=2)
    image_size = (
        f"{int(image_dimensions[0])} x {int(image_dimensions[1])} pixels"
        if image_dimensions is not None
        else "unknown"
    )
    return (
        template.replace("{{REQUEST_JSON}}", request_json)
        .replace("{{SEED_PAGE_JSON}}", seed_page_json)
        .replace("{{IMAGE_DIMENSIONS}}", image_size)
    )


__all__ = [
    "build_extraction_prompt",
    "build_gemini_autocomplete_prompt",
    "build_gemini_autocomplete_request_payload",
    "build_gemini_fill_prompt",
    "build_gemini_fill_request_payload",
    "build_page_prompt_payload",
    "extraction_prompt_template",
    "gemini_autocomplete_prompt_template",
    "gemini_fill_prompt_template",
    "load_prompt_template",
    "resolve_extraction_prompt_path",
    "resolve_gemini_autocomplete_prompt_path",
    "resolve_gemini_fill_prompt_path",
]
