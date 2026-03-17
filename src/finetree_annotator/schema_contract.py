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
    "note_num": '"note_num": "<integer or null>"',
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


def build_custom_extraction_schema_preview(
    *,
    page_meta_keys: Sequence[str] | None = None,
    fact_keys: Sequence[str] | None = None,
) -> str:
    selected_page_meta_keys = _selected_keys(page_meta_keys, PROMPT_PAGE_META_KEYS)
    selected_fact_keys = _selected_keys(fact_keys, PROMPT_FACT_KEYS)

    meta_lines = [_PAGE_META_SCHEMA_LINES[key] for key in selected_page_meta_keys]
    fact_lines = ['"bbox": [<x>, <y>, <w>, <h>]']
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


def build_custom_extraction_prompt_template(
    *,
    page_meta_keys: Sequence[str] | None = None,
    fact_keys: Sequence[str] | None = None,
) -> str:
    selected_page_meta_keys = _selected_keys(page_meta_keys, PROMPT_PAGE_META_KEYS)
    selected_fact_keys = _selected_keys(fact_keys, PROMPT_FACT_KEYS)
    schema_preview = build_custom_extraction_schema_preview(
        page_meta_keys=selected_page_meta_keys,
        fact_keys=selected_fact_keys,
    )
    selected_page_meta = ", ".join(selected_page_meta_keys) if selected_page_meta_keys else "(none)"
    selected_fact = ", ".join(["bbox", *selected_fact_keys]) if selected_fact_keys else "bbox"
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
        "3. `bbox` must use original-image pixel coordinates `[x, y, w, h]` and tightly cover the value text.",
        "4. Preserve value text exactly as printed, including `%`, commas, parentheses, and dash placeholders.",
        "5. Use JSON `null` for missing optional values. Do not emit the string `\"null\"`.",
        "6. Keep UTF-8 Hebrew directly; do not escape it to unicode sequences.",
        "7. Order facts top-to-bottom; within each row use right-to-left for Hebrew pages and left-to-right for English pages.",
    ]
    next_rule_num = 8
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
    page_types = "|".join(PAGE_TYPE_VALUES)
    statement_types = "|".join(STATEMENT_TYPE_VALUES)
    currencies = "|".join(CURRENCY_VALUES)
    scales = "|".join(str(value) for value in SCALE_VALUES)
    value_types = "|".join(VALUE_TYPE_VALUES)
    value_contexts = "|".join(VALUE_CONTEXT_VALUES)
    natural_signs = "|".join(NATURAL_SIGN_VALUES)
    row_roles = "|".join(ROW_ROLE_VALUES)
    period_types = "|".join(PERIOD_TYPE_VALUES)
    path_sources = "|".join(PATH_SOURCE_VALUES)

    return dedent(
        f"""
        You are extracting financial-statement annotations from a single page image into the exact Pydantic schema `PageExtraction`.

        Return ONLY valid JSON.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Return a page-only wrapper with exactly one page.
        Runtime owns document-level metadata, `schema_version`, and `images_dir`.
        Only return `pages` with page `image`, page `meta`, and page `facts`.

        Top-level schema (exact keys only):
        {{
          "pages": [
            {{
              "image": "<string or null>",
              "meta": {{
                "entity_name": "<string or null>",
                "page_num": "<string or null>",
                "page_type": "{page_types}",
                "statement_type": "{statement_types}|null",
                "title": "<string or null>"
              }},
              "facts": [
                {{
                  "bbox": [<x>, <y>, <w>, <h>],
                  "fact_num": <integer >= 1>,
                  "value": "<string>",
                  "equations": [
                    {{
                      "equation": "<string>",
                      "fact_equation": "<string or null>"
                    }}
                  ]|null,
                  "value_type": "{value_types}|null",
                  "value_context": "{value_contexts}|null",
                  "natural_sign": "{natural_signs}|null",
                  "row_role": "{row_roles}",
                  "currency": "{currencies}|null",
                  "scale": {scales}|null,
                  "period_type": "{period_types}|null",
                  "period_start": "<YYYY-MM-DD|null>",
                  "period_end": "<YYYY-MM-DD|null>",
                  "duration_type": "recurrent|null",
                  "recurring_period": "daily|quarterly|monthly|yearly|null",
                  "note_flag": <true|false>,
                  "note_num": "<integer or null>",
                  "note_name": "<string or null>",
                  "path": ["<string>", "..."],
                  "path_source": "{path_sources}|null",
                  "note_ref": "<string or null>",
                  "comment_ref": "<string or null>"
                }}
              ]
            }}
          ]
        }}

        Type/validation rules (must match schema):
        1. `pages` is required. Emit exactly one page inside `pages`.
        2. All listed page `meta` keys are required.
        3. All listed fact keys are required in every fact object, including `bbox` and `fact_num`; nullable fields must still be present with JSON null.
        4. `fact_num` must be contiguous integers starting at 1 and must match the emitted fact order.
        5. `equations` must be a JSON list or null. Use `null` when the fact has no equation.
        6. Each `equations[]` entry must include non-empty `equation`; `fact_equation` may be null.
        7. Do not emit legacy top-level `equation`, `fact_equation`, or `equation_children` keys inside facts.
        8. `path` is a required list of strings (use `[]` when unknown, never `null`).
        9. `note_flag` must be boolean (never null and never `"true"`/`"false"` strings).
        10. `period_start` and `period_end` must be `YYYY-MM-DD` when provided.
        11. `period_type` must be `instant`, `duration`, `expected`, or null. Use `expected` for projected/budgetary values.
        12. `page_type` must be one of: `{page_types}`.
        13. `statement_type` must be one of: `{statement_types}` or null.
        14. `page_type="statements"` for balance sheet / income statement / cash flow / changes in equity / notes / board report / auditor report / statement of activities / other declaration pages.
        15. Non-statement structural pages use `statement_type=null`.
        16. `value` must be a non-empty string. Keep source symbols like `<`, `>`, `(`, `)`, `.`, `*` when present.
        17. `value_type` must be `amount`, `percent`, `ratio`, `count`, `points`, or null.
        18. If `value_type="percent"`, keep `%` inside `value` and set `currency` to null.
        19. `value_context` must be `textual`, `tabular`, `mixed`, or null.
        20. `natural_sign` must be `positive`, `negative`, or null and is derived from `value`:
            - if `value` is wrapped in angle brackets and the inner text is negative like `<-123>` or `<(123)>`, set `natural_sign="negative"`
            - if `value` contains both `(` and `)`, set `natural_sign="negative"`
            - if `value` starts with `-`, set `natural_sign="negative"`
            - if `value` is exactly `"-"`, set `natural_sign=null`
            - otherwise set `natural_sign="positive"`
        21. `row_role` must be `detail` or `total` and indicates whether this row is a detail row or a computed total/subtotal row.
        22. `duration_type` must be `recurrent` or null; set `recurring_period` to `daily`, `quarterly`, `monthly`, or `yearly` when the fact recurs.
        23. `path_source` is only `observed`, `inferred`, or null.
        24. `note_num` must be a JSON integer or `null` only. Never emit a quoted number.
        25. If `note_num` is present, `note_flag` must be `true`.
        26. If `statement_type` is not `notes_to_financial_statements`, all facts must have `note_flag=false` and `note_num=null`.
        27. Use JSON `null` literal for missing optional values (never `"null"` string).
        28. Do not include any keys not listed above.

        Extraction rules:
        1. Extract all visible numeric/table facts, including negatives in parentheses and totals.
        2. Only emit facts anchored on a visible numeric value or numeric symbol cell. Do not emit standalone text labels, section titles, row labels, column headers, page titles, or captions such as `נכסים שוטפים` unless they are part of a numeric fact being extracted.
        3. If source value is exactly `-`, `—`, or `–`, keep `value` as `"-"` (never empty).
        4. Preserve value text as shown, including symbols such as `<`, `>`, `(`, `)`, `.`, and `*`. Do not output empty strings.
        5. For percent values, keep `%` in `value`.
        6. `bbox` must tightly cover the value text only, in pixel coordinates of the original image, as `[x, y, w, h]`.
        7. Assign `fact_num` in the same reading order as emitted facts: top-to-bottom; within a row use right-to-left for Hebrew/RTL pages and left-to-right for English/LTR pages (fallback RTL if uncertain).
        8. `comment_ref`: textual qualifier tied to the specific fact, else `null`.
        9. `note_ref`: use when the fact points or references another note; else `null`.
        10. `note_name`: note title/name only. Do not include generic prefixes such as `Note`, `note`, `באור`, or `ביאור`.
        11. `currency`: infer from page/header context when clear, else `null`.
        12. `scale`: `1` unless page/header indicates thousands or millions.
        13. `value_type`: use `amount` unless the fact is clearly a percent, ratio, count, or points.
        14. `value_context`: choose `textual` for running text, `tabular` for table cells, or `mixed` for hybrid contexts.
        15. Only populate `period_type`, `period_start`, and `period_end` when explicitly known from the page.
        16. `period_type="instant"` means the fact represents a value at a single point in time.
        17. `period_type="duration"` means the fact represents a value over a period between two dates.
        18. `period_type="expected"` marks projected/budgetary values; populate `period_start`/`period_end` only when the future span is defined.
        19. `duration_type="recurrent"` labels a repeating value; populate `recurring_period` with `daily`, `quarterly`, `monthly`, or `yearly` when known.
        20. Use `path_source="observed"` when path labels are directly visible. Use `path_source="inferred"` only when hierarchy is reconstructed from layout/context.
        21. When a total/subtotal has a clearly supported arithmetic expression, populate `equations` with one or more variants. Use `null` when no reliable equation is available.
        22. In each `equations[]` entry, `equation` is the visible arithmetic text and `fact_equation` is the optional fact-reference form like `f2 + f3`.
        23. `natural_sign` is deterministic from `value`: angle-bracketed negatives / parentheses / leading `-` => `negative`, `"-"` => null, otherwise `positive`.
        24. Output UTF-8 Hebrew directly (do not escape to unicode sequences).
        25. Do not emit empty-value facts.
        Page classification rules:
        1. Use `page_type="title"` for cover/title pages.
        2. Use `page_type="contents"` for table-of-contents pages.
        3. Use `page_type="declaration"` for declaration/signoff pages.
        4. Use `page_type="statements"` for business-content statement/report pages.
        5. Use `page_type="other"` only when the page fits none of the above.
        6. Use `statement_type="balance_sheet"` for מאזן.
        7. Use `statement_type="income_statement"` for דוח רווח והפסד / P&L.
        8. Use `statement_type="cash_flow_statement"` for דוח על תזרימי מזומנים.
        9. Use `statement_type="statement_of_changes_in_equity"` for דוח על השינויים בהון העצמי.
        10. Use `statement_type="notes_to_financial_statements"` for באורים לדוחות הכספיים.
        11. Use `statement_type="board_of_directors_report"` for דוח הדירקטוריון.
        12. Use `statement_type="auditors_report"` for דוח רואה החשבון המבקר.
        13. Use `statement_type="statement_of_activities"` for דוח על הפעילויות.
        14. Use `statement_type="other_declaration"` for declaration/report pages that are business declarations but do not fit the other statement types.

        Path hierarchy rule:
        1. Build `path` with horizontal-axis hierarchy first (group -> subgroup -> line item).
        2. Then append vertical-axis hierarchy only when it adds business semantics not already represented by dedicated fields.
        3. Do not encode period/currency/scale/value-format in `path` when already represented by dedicated fields.
        4. Keep labels exactly as shown. Do not invent levels.
        5. Do not include generic note markers like `Note`, `note`, `באור`, or `ביאור` inside `path`.
        """
    ).strip()


def default_gemini_fill_prompt_template() -> str:
    statement_types = "|".join(_PATCH_CONTRACT["statement_types"])
    period_types = "|".join(_PATCH_CONTRACT["period_types"])
    path_sources = "|".join(_PATCH_CONTRACT["path_sources"])
    value_types = "|".join(_PATCH_CONTRACT["value_types"])
    currencies = "|".join(_PATCH_CONTRACT["currencies"])
    scales = "|".join(str(value) for value in _PATCH_CONTRACT["scales"])
    natural_signs = "|".join(NATURAL_SIGN_VALUES)
    row_roles = "|".join(ROW_ROLE_VALUES)

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
              "updates": {{
                "period_type": "{period_types}|null",
                "period_start": "<YYYY-MM-DD|null>",
                "period_end": "<YYYY-MM-DD|null>",
                "duration_type": "recurrent|null",
                "recurring_period": "daily|quarterly|monthly|yearly|null",
                "date": "<YYYY|YYYY-MM|YYYY-MM-DD|null>",
                "value_context": "textual|tabular|mixed|null",
                "value_type": "{value_types}|null",
                "currency": "{currencies}|null",
                "scale": {scales}|null,
                "natural_sign": "{natural_signs}|null",
                "row_role": "{row_roles}",
                "equations": [
                  {{
                    "equation": "<string>",
                    "fact_equation": "<string|null>"
                  }}
                ]|null,
                "path_source": "{path_sources}|null",
                "comment_ref": "<string|null>",
                "note_ref": "<string|null>",
                "note_name": "<string|null>"
              }}
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
        7. `row_role` must be `detail` or `total`.
        8. `equations` must be a JSON list or null. Each `equations[]` entry must include non-empty `equation`; `fact_equation` may be null.
        9. Do not emit legacy top-level `equation`, `fact_equation`, or `equation_children` keys in `updates`.
        10. If no confident update for a requested field, omit that field from `updates`.
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
    "default_gemini_autocomplete_prompt_template",
    "default_gemini_fill_prompt_template",
    "default_extraction_prompt_template",
    "page_level_predicted_schema_document",
    "schema_snapshot",
]
