from __future__ import annotations

from textwrap import dedent
from typing import Any

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
    entity_types = "|".join(ENTITY_TYPE_VALUES)
    period_types = "|".join(PERIOD_TYPE_VALUES)
    report_scope_values = "|".join(REPORT_SCOPE_VALUES)
    path_sources = "|".join(PATH_SOURCE_VALUES)

    return dedent(
        f"""
        You are extracting financial-statement annotations from a single page image into the exact Pydantic schema `PageExtraction`.

        Return ONLY valid JSON.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Return a full document wrapper with exactly one page.
        Runtime owns `schema_version`, and overwrites `images_dir` and `pages[0].image`.
        You must still include `images_dir` and `pages[0].image`.

        Top-level schema (exact keys only):
        {{
          "images_dir": "<string or null>",
          "metadata": {{
            "language": "he|en|null",
            "reading_direction": "rtl|ltr|null",
            "company_name": "<string or null>",
            "company_id": "<string or null>",
            "report_year": "<integer or null>",
            "report_scope": "{report_scope_values}|null",
            "entity_type": "{entity_types}|null"
          }},
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
                  "value": "<string>",
                  "value_type": "{value_types}|null",
                  "value_context": "{value_contexts}|null",
                  "natural_sign": "{natural_signs}|null",
                  "row_role": "{row_roles}",
                  "currency": "{currencies}|null",
                  "scale": {scales}|null,
                  "date": "<YYYY|YYYY-MM|YYYY-MM-DD|null>",
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
        1. `images_dir`, `metadata`, and `pages` are required. Emit exactly one page inside `pages`.
        2. All listed `metadata` keys are required; nullable fields must still be present with JSON null.
        3. All listed page `meta` keys are required.
        4. All listed fact keys are required in every fact object, including `bbox`; nullable fields must still be present with JSON null.
        5. `path` is a required list of strings (use `[]` when unknown, never `null`).
        6. `note_flag` must be boolean (never null and never `"true"`/`"false"` strings).
        7. `date` must be `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` when provided.
        8. `period_start` and `period_end` must be `YYYY-MM-DD` when provided.
        9. `period_type` must be `instant`, `duration`, `expected`, or null. Use `expected` for projected/budgetary values.
        10. `report_scope` must be {report_scope_values} or null.
        11. `page_type` must be one of: `{page_types}`.
        12. `statement_type` must be one of: `{statement_types}` or null.
        13. `page_type="statements"` for balance sheet / income statement / cash flow / changes in equity / notes / board report / auditor report / statement of activities / other declaration pages.
        14. Non-statement structural pages use `statement_type=null`.
        15. `value` must be a non-empty string. Keep source symbols like `<`, `>`, `(`, `)`, `.`, `*` when present.
        16. `value_type` must be `amount`, `percent`, `ratio`, `count`, `points`, or null.
        17. If `value_type="percent"`, keep `%` inside `value` and set `currency` to null.
        18. `value_context` must be `textual`, `tabular`, `mixed`, or null.
        19. `natural_sign` must be `positive`, `negative`, or null and is derived from `value`:
            - if `value` contains both `(` and `)`, set `natural_sign="negative"`
            - if `value` is exactly `"-"`, set `natural_sign=null`
            - otherwise set `natural_sign="positive"`
        20. `row_role` must be `detail` or `total` and indicates whether this row is a detail row or a computed total/subtotal row.
        21. `duration_type` must be `recurrent` or null; set `recurring_period` to `daily`, `quarterly`, `monthly`, or `yearly` when the fact recurs.
        22. `path_source` is only `observed`, `inferred`, or null.
        23. `note_num` must be a JSON integer or `null` only. Never emit a quoted number.
        24. If `note_num` is present, `note_flag` must be `true`.
        25. If `statement_type` is not `notes_to_financial_statements`, all facts must have `note_flag=false` and `note_num=null`.
        26. Use JSON `null` literal for missing optional values (never `"null"` string).
        27. Do not include any keys not listed above.

        Extraction rules:
        1. Extract all visible numeric/table facts, including negatives in parentheses and totals.
        2. Only emit facts anchored on a visible numeric value or numeric symbol cell. Do not emit standalone text labels, section titles, row labels, column headers, page titles, or captions such as `נכסים שוטפים` unless they are part of a numeric fact being extracted.
        3. If source value is exactly `-`, `—`, or `–`, keep `value` as `"-"` (never empty).
        4. Preserve value text as shown, including symbols such as `<`, `>`, `(`, `)`, `.`, and `*`. Do not output empty strings.
        5. For percent values, keep `%` in `value`.
        6. `bbox` must tightly cover the value text only, in pixel coordinates of the original image, as `[x, y, w, h]`.
        7. `comment_ref`: textual qualifier tied to the specific fact, else `null`.
        8. `note_ref`: use when the fact points or references another note; else `null`.
        9. `note_name`: note title/name only. Do not include generic prefixes such as `Note`, `note`, `באור`, or `ביאור`.
        10. `currency`: infer from page/header context when clear, else `null`.
        11. `scale`: `1` unless page/header indicates thousands or millions.
        12. `value_type`: use `amount` unless the fact is clearly a percent, ratio, count, or points.
        13. `value_context`: choose `textual` for running text, `tabular` for table cells, or `mixed` for hybrid contexts.
        14. Preserve `date` when visible. Only populate `period_type`, `period_start`, and `period_end` when explicitly known from the page.
        15. `period_type="instant"` means the fact represents a value at a single point in time.
        16. `period_type="duration"` means the fact represents a value over a period between two dates.
        17. `period_type="expected"` marks projected/budgetary values; populate `period_start`/`period_end` only when the future span is defined.
        18. `duration_type="recurrent"` labels a repeating value; populate `recurring_period` with `daily`, `quarterly`, `monthly`, or `yearly` when known.
        19. Use `path_source="observed"` when path labels are directly visible. Use `path_source="inferred"` only when hierarchy is reconstructed from layout/context.
        20. `natural_sign` is deterministic from `value`: parentheses => `negative`, `"-"` => null, otherwise `positive`.
        21. Order `facts` top-to-bottom; within each row use right-to-left for Hebrew/RTL pages and left-to-right for English/LTR pages (fallback RTL if uncertain).
        22. Output UTF-8 Hebrew directly (do not escape to unicode sequences).
        23. Do not emit empty-value facts.
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
        6. `natural_sign` is deterministic from `value`: parentheses => `negative`, `"-"` => null, otherwise `positive`.
        7. `row_role` must be `detail` or `total`.
        8. If no confident update for a requested field, omit that field from `updates`.
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

        Locked snapshot JSON:
        {{{{REQUEST_JSON}}}}

        start with this

        {{{{SEED_PAGE_JSON}}}}

        Auto Complete constraints:
        1. Use the exact extraction schema contract below for output shape and field validation.
        2. Return the same full wrapper shape as extraction (`images_dir`, `metadata`, `pages` with one page).
        3. Include only new missing facts in `pages[0].facts`; never include locked facts from the snapshot in output.
        4. Keep `metadata` and `pages[0].meta` aligned with the locked snapshot.
        5. `bbox` values must be in original image pixel coordinates.
        6. Do not rely on locked `fact_num` values while generating; runtime rebuilds final contiguous numbering after merge.
        7. If there are no confidently missing facts, return an empty `facts` array.

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
    "schema_snapshot",
]
