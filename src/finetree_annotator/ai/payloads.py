from __future__ import annotations

import json
import os
from pathlib import Path
import re
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
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_ROOT / "system_prompt.txt"
LEGACY_EXTRACTION_PROMPT_PATH = REPO_ROOT / "prompt.txt"
DEFAULT_SYSTEM_PROMPT_TEXT = (
    "You are a precise financial statement extraction system. "
    "Return only valid JSON that matches the required schema."
)
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


def resolve_system_prompt_path() -> Optional[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("FINETREE_SYSTEM_PROMPT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([Path.cwd() / "prompts" / "system_prompt.txt", DEFAULT_SYSTEM_PROMPT_PATH, Path.cwd() / "system_prompt.txt"])
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "prompts" / "system_prompt.txt")
        candidates.append(parent / "system_prompt.txt")
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


def system_prompt_template() -> str:
    return load_prompt_template(resolve_system_prompt_path(), fallback_text=DEFAULT_SYSTEM_PROMPT_TEXT).strip()


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


def build_gemini_bbox_only_prompt(
    page_image_path: Path,
    *,
    image_dimensions: Optional[tuple[float, float]],
) -> str:
    base_prompt = build_extraction_prompt(
        extraction_prompt_template(),
        page_image_path,
        image_dimensions=image_dimensions,
    )
    prompt = _bbox_only_prompt_from_extraction_prompt(base_prompt)
    bbox_only_rules = (
        "BBox+value mode override:\n"
        "- Keep the same overall JSON wrapper shape.\n"
        "- For each fact, return only `bbox` and `value`.\n"
        "- Do not include any other fact fields in this mode.\n"
        "- Include all numerical entities that function as numeric cells/values.\n"
        "- Include standalone \"-\" entries when they function as numeric placeholders/values.\n"
        "- `bbox` must tightly cover the visible text only, in pixel coordinates of the original image.\n\n"
        "In this mode, each fact object must have exactly this shape:\n"
        "{\"bbox\": [x, y, w, h], \"value\": \"<string>\"}\n"
    )
    return f"{prompt.rstrip()}\n\n{bbox_only_rules}"


def build_qwen_bbox_only_prompt(
    page_image_path: Path,
    *,
    image_dimensions: Optional[tuple[float, float]],
    prepared_dimensions: Optional[tuple[int, int]],
    max_pixels: int,
) -> str:
    base_prompt = build_extraction_prompt(
        extraction_prompt_template(),
        page_image_path,
        image_dimensions=image_dimensions,
    )
    prompt = _bbox_only_prompt_from_extraction_prompt(base_prompt)
    prompt = prompt.replace(
        "4. `bbox` must tightly cover the value text only, in original image pixel coordinates `[x, y, w, h]`.",
        "4. `bbox` must tightly cover the value text only, in prepared image pixel coordinates `[x, y, w, h]`.",
    )
    original_size = (
        f"{int(image_dimensions[0])} x {int(image_dimensions[1])} pixels"
        if image_dimensions is not None
        else "unknown"
    )
    prepared_size = (
        f"{int(prepared_dimensions[0])} x {int(prepared_dimensions[1])} pixels"
        if prepared_dimensions is not None
        else "unknown"
    )
    bbox_only_rules = (
        "Qwen bbox+value mode override:\n"
        f"- The original page image size is {original_size}.\n"
        f"- The prepared image sent to Qwen is {prepared_size}.\n"
        f"- The prepared image was resized under a max pixel budget of {int(max_pixels)}.\n"
        "- Keep the same overall JSON wrapper shape.\n"
        "- For each fact, return only `bbox` and `value`.\n"
        "- Do not include any other fact fields in this mode.\n"
        "- Include all numerical entities that function as numeric cells/values.\n"
        "- Include standalone \"-\" entries when they function as numeric placeholders/values.\n"
        "- Return bbox coordinates in prepared-image pixels, not original-image pixels.\n"
        "- Each bbox must tightly cover the visible value text only.\n\n"
        "In this mode, each fact object must have exactly this shape:\n"
        "{\"bbox\": [x, y, w, h], \"value\": \"<string>\"}\n"
    )
    return f"{prompt.rstrip()}\n\n{bbox_only_rules}"


def _bbox_only_prompt_from_extraction_prompt(base_prompt: str) -> str:
    prompt = re.sub(
        r'      "facts": \[\n        \{\n.*?\n        \}\n      \]',
        '      "facts": [\n        {\n  "bbox": [<x>, <y>, <w>, <h>],\n  "value": "<string>"\n        }\n      ]',
        base_prompt,
        count=1,
        flags=re.S,
    )
    return prompt.replace(
        "2. Include all listed `meta` keys and all listed fact keys in every emitted fact. Use JSON `null` for missing optional values.",
        "2. Include all listed `meta` keys. In this mode, each emitted fact must include only `bbox` and `value`.",
    )


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


def build_gemini_fix_drawn_request_payload(
    *,
    page_name: str,
    page_meta: dict[str, Any],
    ordered_fact_payloads: list[dict[str, Any]],
    hand_drawn_fact_nums: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    hand_drawn_set = set(hand_drawn_fact_nums)
    facts_out: list[dict[str, Any]] = []
    for payload in ordered_fact_payloads:
        fact_num = payload.get("fact_num")
        if fact_num in hand_drawn_set:
            facts_out.append({
                "fact_num": fact_num,
                "bbox": payload.get("bbox"),
                **{k: None for k in payload if k not in ("fact_num", "bbox")},
            })
        else:
            facts_out.append({
                "fact_num": payload.get("fact_num"),
                "bbox": payload.get("bbox"),
                **{k: v for k, v in payload.items() if k not in ("fact_num", "bbox")},
            })
    return build_page_prompt_payload(page_name=page_name, page_meta=dict(page_meta or {}), facts=facts_out)


def build_gemini_fix_drawn_prompt(
    *,
    request_payload: dict[str, Any],
    hand_drawn_fact_nums: tuple[int, ...] | list[int],
    image_dimensions: Optional[tuple[float, float]],
) -> str:
    request_json = serialize_schema_mode_payload(request_payload)
    drawn_nums_str = ", ".join(str(n) for n in hand_drawn_fact_nums)
    image_size = (
        f"{int(image_dimensions[0])} x {int(image_dimensions[1])} pixels"
        if image_dimensions is not None
        else "unknown"
    )
    return (
        "You are completing all annotation fields for newly-drawn bounding boxes on a financial page.\n\n"
        "You receive:\n"
        "1. The page image.\n"
        "2. A JSON snapshot of ALL facts on the page.\n"
        "   Facts with all-null fields (except bbox) are the newly-drawn ones that need completion.\n"
        f"3. The fact_num values of drawn facts that need completion: {drawn_nums_str}.\n\n"
        f"Current image size: {image_size}.\n\n"
        "Existing (non-drawn) facts are authoritative context.\n"
        "Use their patterns (path structure, periods, currencies, scales, value_types, etc.) "
        "to infer fields for the drawn facts.\n\n"
        f"Input snapshot JSON:\n{request_json}\n\n"
        "Return ONLY valid JSON matching this exact schema:\n"
        "{\n"
        '  "meta_updates": {},\n'
        '  "fact_updates": [\n'
        "    {\n"
        '      "fact_num": <integer matching a drawn fact>,\n'
        '      "updates": {\n'
        '        "value": "<string — read from image within the bbox area>",\n'
        '        "path": ["<string>", "..."],\n'
        '        "row_role": "detail|total",\n'
        '        "natural_sign": "positive|negative|null",\n'
        '        "value_type": "amount|percent|ratio|count|points|null",\n'
        '        "value_context": "textual|tabular|mixed|null",\n'
        '        "currency": "ILS|USD|EUR|GBP|null",\n'
        '        "scale": 1|1000|1000000|null,\n'
        '        "period_type": "instant|duration|expected|null",\n'
        '        "period_start": "YYYY-MM-DD|null",\n'
        '        "period_end": "YYYY-MM-DD|null",\n'
        '        "duration_type": "recurrent|null",\n'
        '        "recurring_period": "daily|quarterly|monthly|yearly|null",\n'
        '        "note_flag": true|false,\n'
        '        "note_name": "<string or null>",\n'
        '        "note_num": "<integer or null>",\n'
        '        "note_ref": "<string or null>",\n'
        '        "comment_ref": "<string or null>",\n'
        '        "path_source": "observed|inferred|null"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. Return patch-only JSON. No markdown, no code fences, no prose.\n"
        f"2. Only return updates for the drawn facts (fact_nums: {drawn_nums_str}). Never update non-drawn facts.\n"
        "3. Fill ALL annotation fields for each drawn fact. Read the value from the image within the drawn bbox.\n"
        "4. Use existing facts as patterns for path, period, currency, scale, etc.\n"
        "5. Do not include bbox in updates — it is already correct.\n"
        "6. Keep value exactly as printed in the image within the bbox area.\n"
        '7. natural_sign: parentheses / angle brackets / leading "-" => "negative", standalone "-" => null, otherwise "positive".\n'
        '8. row_role must be "detail" or "total".\n'
        "9. Use JSON null (not the string \"null\") for unknowns.\n"
        '10. meta_updates must be {} (no meta changes in this mode).\n'
    )


def build_auto_annotate_enrich_seed_payload(
    *,
    page_name: str,
    page_meta: dict[str, Any],
    ordered_fact_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the seed input payload for Auto-Annotate enrich: each fact has only {fact_num, bbox, value}."""
    seed_facts = [
        {
            "fact_num": i,
            "bbox": payload.get("bbox"),
            "value": payload.get("value"),
        }
        for i, payload in enumerate(ordered_fact_payloads, start=1)
    ]
    return build_page_prompt_payload(page_name=page_name, page_meta=page_meta, facts=seed_facts)


def build_auto_annotate_enrich_prompt(
    *,
    seed_payload: dict[str, Any],
    image_dimensions: Optional[tuple[float, float]],
) -> str:
    """Build the prompt for Auto-Annotate enrich mode.

    Instructs Gemini to fill in all annotation fields for pre-detected facts,
    preserving the given bbox and value exactly.
    """
    seed_json = serialize_schema_mode_payload(seed_payload)
    image_size = (
        f"{int(image_dimensions[0])} x {int(image_dimensions[1])} pixels"
        if image_dimensions is not None
        else "unknown"
    )
    return (
        "Auto-Annotate enrich mode.\n"
        f"Image size: {image_size}.\n\n"
        "You are given existing numeric detections (fact_num, bbox, value) for this financial page.\n"
        "For each fact:\n"
        "- Preserve the given bbox and value exactly as provided.\n"
        "- Fill in all remaining annotation fields: path, row_role, value_type, value_context,\n"
        "  currency, scale, natural_sign, note_flag, note_name, note_num, note_ref, comment_ref,\n"
        "  period_type, period_start, period_end, duration_type, recurring_period, path_source.\n"
        "Also fill in the page meta fields (entity_name, page_num, page_type, statement_type, title).\n"
        "Return facts in the same order as the input (by fact_num).\n\n"
        "Existing detections:\n"
        f"{seed_json}"
    )


__all__ = [
    "build_auto_annotate_enrich_prompt",
    "build_auto_annotate_enrich_seed_payload",
    "build_extraction_prompt",
    "build_gemini_autocomplete_prompt",
    "build_gemini_bbox_only_prompt",
    "build_gemini_autocomplete_request_payload",
    "build_gemini_fill_prompt",
    "build_gemini_fill_request_payload",
    "build_gemini_fix_drawn_prompt",
    "build_gemini_fix_drawn_request_payload",
    "build_page_prompt_payload",
    "build_qwen_bbox_only_prompt",
    "extraction_prompt_template",
    "gemini_autocomplete_prompt_template",
    "gemini_fill_prompt_template",
    "load_prompt_template",
    "resolve_extraction_prompt_path",
    "resolve_gemini_autocomplete_prompt_path",
    "resolve_gemini_fill_prompt_path",
]
