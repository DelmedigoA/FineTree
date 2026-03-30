from __future__ import annotations

from textwrap import dedent
from typing import Any, Sequence

from .schema_registry import SchemaRegistry

_EXTRACT_CONTRACT = SchemaRegistry.get_prompt_contract("extraction")
_PATCH_CONTRACT = SchemaRegistry.get_prompt_contract("gemini_fill")
_PAGE_META_MODEL_SPEC = SchemaRegistry.get_model_spec("page_meta")
_FACT_MODEL_SPEC = SchemaRegistry.get_model_spec("fact")
PROMPT_TOP_LEVEL_KEYS: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["top_level_keys"])
METADATA_KEYS: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["metadata_keys"])
PROMPT_PAGE_META_KEYS: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["page_meta_keys"])
PAGE_META_KEYS: tuple[str, ...] = tuple(_PAGE_META_MODEL_SPEC.canonical_write_keys)
PROMPT_FACT_KEYS: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["fact_keys"])
CANONICAL_FACT_KEYS: tuple[str, ...] = tuple(_FACT_MODEL_SPEC.canonical_write_keys)
EXTRACTED_FACT_KEYS: tuple[str, ...] = (*CANONICAL_FACT_KEYS, "bbox")
REQUIRED_PROMPT_CANONICAL_KEYS: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["required_prompt_fact_keys"])
LEGACY_FACT_KEYS: tuple[str, ...] = tuple(_FACT_MODEL_SPEC.read_alias_keys)
PAGE_TYPE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["page_types"])
STATEMENT_TYPE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["statement_types"])
VALUE_TYPE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["value_types"])
VALUE_CONTEXT_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["value_contexts"])
NATURAL_SIGN_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["natural_signs"])
ROW_ROLE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["row_roles"])
CURRENCY_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["currencies"])
SCALE_VALUES: tuple[int, ...] = tuple(_EXTRACT_CONTRACT["enums"]["scales"])
ENTITY_TYPE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["entity_types"])
PERIOD_TYPE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["period_types"])
PATH_SOURCE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["path_sources"])
REPORT_SCOPE_VALUES: tuple[str, ...] = tuple(_EXTRACT_CONTRACT["enums"]["report_scope"])

_PAGE_META_SCHEMA_LINES: dict[str, str] = {
    "entity_name": '"entity_name": "<string or null>"',
    "page_num": '"page_num": "<string or null>"',
    "page_type": f'"page_type": "{"|".join(PAGE_TYPE_VALUES)}"',
    "statement_type": f'"statement_type": "{"|".join(STATEMENT_TYPE_VALUES)}|null"',
    "title": '"title": "<string or null>"',
}

_FACT_SCHEMA_LINES: dict[str, str] = {
    "fact_num": '"fact_num": <integer >= 1>',
    "value": '"value": "<string>"',
    "equations": dedent(
        """
        "equations": [
          {
            "equation": "<string>",
            "fact_equation": "<string or null>"
          }
        ]|null
        """
    ).strip(),
    "value_type": f'"value_type": "{"|".join(VALUE_TYPE_VALUES)}|null"',
    "value_context": f'"value_context": "{"|".join(VALUE_CONTEXT_VALUES)}|null"',
    "natural_sign": f'"natural_sign": "{"|".join(NATURAL_SIGN_VALUES)}|null"',
    "row_role": f'"row_role": "{"|".join(ROW_ROLE_VALUES)}"',
    "currency": f'"currency": "{"|".join(CURRENCY_VALUES)}|null"',
    "scale": f'"scale": {"|".join(str(value) for value in SCALE_VALUES)}|null',
    "date": '"date": "<YYYY|YYYY-MM|YYYY-MM-DD|null>"',
    "period_type": f'"period_type": "{"|".join(PERIOD_TYPE_VALUES)}|null"',
    "period_start": '"period_start": "<YYYY-MM-DD|null>"',
    "period_end": '"period_end": "<YYYY-MM-DD|null>"',
    "duration_type": '"duration_type": "recurrent|null"',
    "recurring_period": '"recurring_period": "daily|quarterly|monthly|yearly|null"',
    "note_flag": '"note_flag": <true|false>',
    "note_num": '"note_num": "<string or null>"',
    "note_name": '"note_name": "<string or null>"',
    "path": '"path": ["<string>", "..."]',
    "path_source": f'"path_source": "{"|".join(PATH_SOURCE_VALUES)}|null"',
    "note_ref": '"note_ref": "<string or null>"',
    "comment_ref": '"comment_ref": "<string or null>"',
}


def _selected_keys(keys: Sequence[str] | None, allowed: Sequence[str]) -> list[str]:
    allowed_set = {str(key) for key in allowed}
    if keys is None:
        return [str(key) for key in allowed]
    selected: list[str] = []
    for key in keys:
        normalized = str(key or "").strip()
        if not normalized or normalized not in allowed_set or normalized in selected:
            continue
        selected.append(normalized)
    return selected


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else line for line in str(text).splitlines())


def _patch_fact_schema_lines() -> list[str]:
    return [_FACT_SCHEMA_LINES[key] for key in _PATCH_CONTRACT["fact_patch_fields"]]


def build_gemini_fill_updates_schema(selected_fact_fields: Sequence[str] | None = None) -> str:
    selected_patch_keys = _selected_keys(selected_fact_fields, _PATCH_CONTRACT["fact_patch_fields"])
    if not selected_patch_keys:
        return "{}"
    schema_lines = ",\n".join(_FACT_SCHEMA_LINES[key] for key in selected_patch_keys)
    return "{\n" + _indent_block(schema_lines, "  ") + "\n}"


def build_custom_extraction_schema_preview(
    *,
    page_meta_keys: Sequence[str] | None = None,
    fact_keys: Sequence[str] | None = None,
    include_bbox: bool = True,
    bbox_schema_line: str = '"bbox": [<x>, <y>, <w>, <h>]',
) -> str:
    selected_page_meta_keys = _selected_keys(page_meta_keys, PROMPT_PAGE_META_KEYS)
    selected_fact_keys = _selected_keys(fact_keys, PROMPT_FACT_KEYS)

    meta_lines = [_PAGE_META_SCHEMA_LINES[key] for key in selected_page_meta_keys]
    fact_lines: list[str] = []
    if include_bbox:
        fact_lines.append(str(bbox_schema_line))
    fact_lines.extend(_FACT_SCHEMA_LINES[key] for key in selected_fact_keys)
    meta_body = ",\n        ".join(meta_lines)
    fact_body = ",\n      ".join(fact_lines)
    return dedent(
        f"""
        {{
          "meta": {{
            {meta_body}
          }},
          "facts": [
            {{
              {fact_body}
            }}
          ]
        }}
        """
    ).strip()


def build_gemini_bbox_page_schema_preview(
    *,
    page_meta_keys: Sequence[str] | None = None,
    fact_keys: Sequence[str] | None = None,
) -> str:
    return build_custom_extraction_schema_preview(
        page_meta_keys=page_meta_keys,
        fact_keys=fact_keys,
        include_bbox=True,
        bbox_schema_line='"bbox": [<ymin>, <xmin>, <ymax>, <xmax>]',
    )


def build_custom_extraction_prompt_template(
    *,
    page_meta_keys: Sequence[str] | None = None,
    fact_keys: Sequence[str] | None = None,
    include_bbox: bool = True,
) -> str:
    selected_page_meta_keys = _selected_keys(page_meta_keys, PROMPT_PAGE_META_KEYS)
    selected_fact_keys = _selected_keys(fact_keys, PROMPT_FACT_KEYS)
    schema_preview = build_custom_extraction_schema_preview(
        page_meta_keys=selected_page_meta_keys,
        fact_keys=selected_fact_keys,
        include_bbox=include_bbox,
    )
    selected_page_meta = ", ".join(selected_page_meta_keys) if selected_page_meta_keys else "(none)"
    selected_fact_names = list(selected_fact_keys)
    if include_bbox:
        selected_fact_names = ["bbox", *selected_fact_names]
    selected_fact = ", ".join(selected_fact_names) if selected_fact_names else "(none)"
    lines = [
        "You are extracting financial-statement annotations from a single page image.",
        "",
        "Return ONLY valid JSON.",
        "Do NOT return markdown, code fences, comments, prose, or extra keys.",
        "",
        "Return the exact page-level object shown below. Include only the selected page-meta and fact keys.",
        "",
        "Selected page meta keys:",
        f"- {selected_page_meta}",
        "",
        "Selected fact keys:",
        f"- {selected_fact}",
        "",
        "Exact schema:",
        schema_preview,
        "",
        "Rules:",
        "1. Return only a single page-level object with `meta` and `facts`.",
        "2. Extract only visible numeric/table facts. Do not emit standalone labels or headings as facts.",
    ]
    next_rule_num = 3
    if include_bbox:
        lines.append(
            f"{next_rule_num}. `bbox` must use original-image pixel coordinates `[x, y, w, h]` and tightly cover the value text."
        )
        next_rule_num += 1
    lines.extend(
        [
            f"{next_rule_num}. Preserve value text exactly as printed, including `%`, commas, parentheses, and dash placeholders.",
            f"{next_rule_num + 1}. Use JSON `null` for missing optional values. Do not emit the string `\"null\"`.",
            f"{next_rule_num + 2}. Keep UTF-8 Hebrew directly; do not escape it to unicode sequences.",
            f"{next_rule_num + 3}. Order facts top-to-bottom; within each row use right-to-left for Hebrew pages and left-to-right for English pages.",
        ]
    )
    next_rule_num += 4
    if "fact_num" in selected_fact_keys:
        lines.append(f"{next_rule_num}. `fact_num` must be contiguous integers starting at 1 and must match the emitted fact order.")
        next_rule_num += 1
    if "equations" in selected_fact_keys:
        lines.append(f"{next_rule_num}. Use `equations=null` unless the source explicitly shows a reliable arithmetic relation.")
        next_rule_num += 1
    if "path" in selected_fact_keys:
        lines.append(f"{next_rule_num}. Keep `path` as a list of visible hierarchy labels; use `[]` when unknown.")
        next_rule_num += 1
    if "page_type" in selected_page_meta_keys or "statement_type" in selected_page_meta_keys:
        lines.append(f"{next_rule_num}. Classify page type and statement type from visible page context only.")
    return "\n".join(lines).strip()


def page_level_predicted_schema_document() -> str:
    schema_preview = build_custom_extraction_schema_preview(
        page_meta_keys=PROMPT_PAGE_META_KEYS,
        fact_keys=PROMPT_FACT_KEYS,
    )
    page_meta_keys = "\n".join(f"- `{key}`" for key in PROMPT_PAGE_META_KEYS)
    fact_keys = "\n".join(f"- `{key}`" for key in ("bbox", *PROMPT_FACT_KEYS))
    return (
        "# Page Level Predicted Schema\n\n"
        "This file is generated from `src/finetree_annotator/schema_contract.py`.\n"
        "Do not edit it manually.\n\n"
        "Refresh command:\n"
        "```bash\n"
        "PYTHONPATH=src ./.env/bin/python scripts/sync_page_level_predicted_schema.py\n"
        "```\n\n"
        f"Schema version: `{_EXTRACT_CONTRACT['schema_version']}`\n\n"
        "Selected page meta keys:\n"
        f"{page_meta_keys}\n\n"
        "Selected fact keys:\n"
        f"{fact_keys}\n\n"
        "Exact page-level schema:\n"
        "```jsonc\n"
        f"{schema_preview}\n"
        "```\n"
    )


def schema_snapshot() -> dict[str, Any]:
    return {
        "schema_version": _EXTRACT_CONTRACT["schema_version"],
        "page_extraction_keys": list(PROMPT_TOP_LEVEL_KEYS),
        "metadata_keys": list(METADATA_KEYS),
        "page_meta_keys": list(PAGE_META_KEYS),
        "prompt_page_meta_keys": list(PROMPT_PAGE_META_KEYS),
        "extracted_fact_keys": list(EXTRACTED_FACT_KEYS),
        "canonical_fact_keys": list(CANONICAL_FACT_KEYS),
        "prompt_fact_keys": list(PROMPT_FACT_KEYS),
        "required_prompt_fact_keys": list(REQUIRED_PROMPT_CANONICAL_KEYS),
        "legacy_fact_keys": list(LEGACY_FACT_KEYS),
        "page_types": list(PAGE_TYPE_VALUES),
        "statement_types": list(STATEMENT_TYPE_VALUES),
        "value_types": list(VALUE_TYPE_VALUES),
        "value_contexts": list(VALUE_CONTEXT_VALUES),
        "natural_signs": list(NATURAL_SIGN_VALUES),
        "row_role_values": list(ROW_ROLE_VALUES),
        "currencies": list(CURRENCY_VALUES),
        "scales": list(SCALE_VALUES),
        "entity_types": list(ENTITY_TYPE_VALUES),
        "period_types": list(PERIOD_TYPE_VALUES),
        "path_sources": list(PATH_SOURCE_VALUES),
        "report_scope_values": list(REPORT_SCOPE_VALUES),
    }


def default_extraction_prompt_template() -> str:
    meta_lines = ",\n".join(_PAGE_META_SCHEMA_LINES[key] for key in PROMPT_PAGE_META_KEYS)
    fact_lines = ['"bbox": [<x>, <y>, <w>, <h>]']
    fact_lines.extend(_FACT_SCHEMA_LINES[key] for key in PROMPT_FACT_KEYS)
    meta_body = _indent_block(meta_lines, "        ")
    fact_body = _indent_block(",\n".join(fact_lines), "          ")

    return dedent(
        f"""
        You are extracting financial-statement annotations from a single page image into the exact Pydantic schema `PageExtraction`.

        Current page image: {{{{IMAGE_NAME}}}}.
        Current image size: {{{{IMAGE_DIMENSIONS}}}}.

        Return ONLY valid JSON.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Only return `pages` with page `image`, page `meta`, and page `facts`.

        Return exactly this wrapper shape:
        {{
          "pages": [
            {{
              "image": "<string or null>",
              "meta": {{
{meta_body}
              }},
              "facts": [
                {{
{fact_body}
                }}
              ]
            }}
          ]
        }}

        1. `pages` is required. Emit exactly one page inside `pages`.
        2. Include all listed `meta` keys and all listed fact keys in every emitted fact. Use JSON `null` for missing optional values.
        3. Only emit facts anchored on a visible numeric value or numeric symbol cell. Do not emit standalone labels, section titles, row labels, column headers, page titles, or captions such as `נכסים שוטפים`.
        4. `bbox` must tightly cover the value text only, in original image pixel coordinates `[x, y, w, h]`.
        5. `fact_num` must be contiguous integers starting at 1 and must match the emitted fact order.
        6. Order facts top-to-bottom; within each row use right-to-left for Hebrew/RTL pages and left-to-right for English/LTR pages.
        7. Keep `value` exactly as printed, including `%`, commas, parentheses, leading `-`, angle brackets, and dash placeholders. If the source cell is `-`, `—`, or `–`, return `"-"`.
        8. `path` must be a JSON list of visible hierarchy labels. Use `[]` when unknown.
        9. If `comment_ref` seems unreasonably long, do not include the full text. Use a short marker only, for example `"*"` or `"(1)"`.
        10. `equations` must be a JSON list or `null`. Use `null` unless the page visibly supports a reliable arithmetic relation.
        11. Classify `page_type` and `statement_type` from visible page context only.
        12. Do not emit legacy top-level `equation`, `fact_equation`, or `equation_children` keys inside facts.
        13. Do not emit runtime-only page keys such as `annotation_note` or `annotation_status`, and do not emit `date`.
        14. Extract a fact only when you can localize its numeric cell tightly. Skip uncertain or non-localizable facts.
        15. Output UTF-8 Hebrew directly. Do not escape it to unicode sequences.
        """
    ).strip()


def default_gemini_fill_prompt_template() -> str:
    statement_types = "|".join(_PATCH_CONTRACT["statement_types"])

    return dedent(
        f"""
        You are updating selected fields for already-annotated financial facts on a single page.

        You receive:
        1. The page image.
        2. A redacted JSON snapshot where requested target fields are set to null.

        Return ONLY valid JSON patch output.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Important semantics:
        - `instant` means the fact represents a value at a single point in time.
        - `duration` means the fact represents a value over a period between two dates.
        - `expected` indicates a projected or budgeted value; include future periods if available.
        - `duration_type="recurrent"` means the value repeats; set `recurring_period` to `daily`, `quarterly`, `monthly`, or `yearly` when known.

        Requested fact fields:
        {{{{FACT_FIELDS}}}}

        Requested meta fields:
        {{{{META_FIELDS}}}}

        Input snapshot JSON:
        {{{{REQUEST_JSON}}}}

        Output schema (exact keys only):
        {{
          "meta_updates": {{{{META_UPDATES_SCHEMA}}}},
          "fact_updates": [
            {{
              "fact_num": <integer >= 1>,
              "updates": {{{{FACT_UPDATES_SCHEMA}}}}
            }}
          ]
        }}

        Rules:
        1. Return patch-only JSON. Never return full page annotations.
        2. Update only requested fields. Do not add non-requested fields.
        2a. If no meta fields were requested, return `"meta_updates": {{}}`.
        3. `fact_num` must match the provided snapshot identity.
        4. Keep `fact_updates` focused and minimal. Include only facts that need updates.
        5. Use JSON null (not string "null") for unknowns.
        6. `natural_sign` is deterministic from `value`: angle-bracketed negatives / parentheses / leading `-` => `negative`, `"-"` => null, otherwise `positive`.
        6a. If the visible source value is `-`, `—`, or `–`, set `value` to `"-"`. Never convert dash placeholders to `0`.
        7. `row_role` must be `detail` or `total`.
        8. `equations` must be a JSON list or null. Each `equations[]` entry must include non-empty `equation`; `fact_equation` may be null.
        9. If `comment_ref` seems unreasonably long, do not include the full text. Use a short marker only, for example `"*"` or `"(1)"`.
        10. Do not emit legacy top-level `equation`, `fact_equation`, or `equation_children` keys in `updates`.
        11. If no confident update for a requested field, omit that field from `updates`.
        """
    ).strip()


def default_gemini_autocomplete_prompt_template() -> str:
    extraction_contract = default_extraction_prompt_template()
    return dedent(
        f"""
        You are completing missing financial-statement annotations for a single page.

        You receive:
        1. The page image.
        2. A JSON snapshot of the current page annotations.
        3. Existing page facts that are already reviewed and must stay locked.

        Return ONLY valid JSON.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Existing facts in the input snapshot are authoritative locked facts.
        Never modify, replace, remove, rename, or re-output locked facts from the snapshot.
        Return only NEW missing facts for the same page.
        Do not emit page-level meta changes in this mode; keep page meta aligned with the locked snapshot.
        New facts will be merged into the page by runtime using bbox geometry and then renumbered by runtime.
        Current image size: {{{{IMAGE_DIMENSIONS}}}}.

        Locked snapshot JSON:
        {{{{REQUEST_JSON}}}}

        start with this

        {{{{SEED_PAGE_JSON}}}}

        Auto Complete constraints:
        1. Use the exact extraction schema contract below for output shape and field validation.
        2. Return the same page-only wrapper shape as extraction (`pages` with one page).
        3. Include only new missing facts in `pages[0].facts`; never include locked facts from the snapshot in output.
        4. Keep `pages[0].image` and `pages[0].meta` aligned with the locked snapshot.
        5. `bbox` values must be in original image pixel coordinates.
        6. Assign `fact_num` contiguously starting at 1 within the new facts you emit. Do not reuse locked snapshot fact numbers.
        7. Runtime rebuilds final contiguous numbering after merge, so emitted `fact_num` is local to the new facts only.
        8. If there are no confidently missing facts, return an empty `facts` array.

        Extraction contract:
        {extraction_contract}
        """
    ).strip()


__all__ = [
    "CANONICAL_FACT_KEYS",
    "CURRENCY_VALUES",
    "ENTITY_TYPE_VALUES",
    "EXTRACTED_FACT_KEYS",
    "LEGACY_FACT_KEYS",
    "METADATA_KEYS",
    "PAGE_META_KEYS",
    "PAGE_TYPE_VALUES",
    "PATH_SOURCE_VALUES",
    "PERIOD_TYPE_VALUES",
    "PROMPT_PAGE_META_KEYS",
    "PROMPT_TOP_LEVEL_KEYS",
    "PROMPT_FACT_KEYS",
    "REQUIRED_PROMPT_CANONICAL_KEYS",
    "SCALE_VALUES",
    "STATEMENT_TYPE_VALUES",
    "NATURAL_SIGN_VALUES",
    "ROW_ROLE_VALUES",
    "VALUE_TYPE_VALUES",
    "VALUE_CONTEXT_VALUES",
    "build_gemini_bbox_page_schema_preview",
    "build_gemini_fill_updates_schema",
    "default_gemini_autocomplete_prompt_template",
    "default_gemini_fill_prompt_template",
    "default_extraction_prompt_template",
    "page_level_predicted_schema_document",
    "schema_snapshot",
]
