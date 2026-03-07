from __future__ import annotations

from textwrap import dedent
from typing import Any

from .schemas import (
    Currency,
    EntityType,
    ExtractedFact,
    Metadata,
    PageExtraction,
    PageMeta,
    PageType,
    PathSource,
    PeriodType,
    Scale,
    StatementType,
    ValueType,
)

PROMPT_TOP_LEVEL_KEYS: tuple[str, ...] = tuple(PageExtraction.model_fields.keys())
METADATA_KEYS: tuple[str, ...] = tuple(Metadata.model_fields.keys())
PAGE_META_KEYS: tuple[str, ...] = tuple(PageMeta.model_fields.keys())
EXTRACTED_FACT_KEYS: tuple[str, ...] = tuple(ExtractedFact.model_fields.keys())
CANONICAL_FACT_KEYS: tuple[str, ...] = tuple(key for key in EXTRACTED_FACT_KEYS if key != "bbox")
REQUIRED_PROMPT_CANONICAL_KEYS: tuple[str, ...] = (
    "comment_ref",
    "note_flag",
    "note_name",
    "note_num",
    "note_ref",
    "period_type",
    "period_start",
    "period_end",
    "path_source",
)
LEGACY_FACT_KEYS: tuple[str, ...] = (
    "ref_comment",
    "comment",
    "ref_note",
    "note_reference",
    "refference",
    "reference",
    "ref",
    "is_beur",
    "beur_num",
    "beur_number",
    "footnote",
)
PAGE_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in PageType)
STATEMENT_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in StatementType)
VALUE_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in ValueType)
CURRENCY_VALUES: tuple[str, ...] = tuple(item.value for item in Currency)
SCALE_VALUES: tuple[int, ...] = tuple(int(item.value) for item in Scale)
ENTITY_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in EntityType)
PERIOD_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in PeriodType)
PATH_SOURCE_VALUES: tuple[str, ...] = tuple(item.value for item in PathSource)


def schema_snapshot() -> dict[str, Any]:
    return {
        "page_extraction_keys": list(PROMPT_TOP_LEVEL_KEYS),
        "metadata_keys": list(METADATA_KEYS),
        "page_meta_keys": list(PAGE_META_KEYS),
        "extracted_fact_keys": list(EXTRACTED_FACT_KEYS),
        "canonical_fact_keys": list(CANONICAL_FACT_KEYS),
        "required_prompt_fact_keys": list(REQUIRED_PROMPT_CANONICAL_KEYS),
        "legacy_fact_keys": list(LEGACY_FACT_KEYS),
        "page_types": list(PAGE_TYPE_VALUES),
        "statement_types": list(STATEMENT_TYPE_VALUES),
        "value_types": list(VALUE_TYPE_VALUES),
        "currencies": list(CURRENCY_VALUES),
        "scales": list(SCALE_VALUES),
        "entity_types": list(ENTITY_TYPE_VALUES),
        "period_types": list(PERIOD_TYPE_VALUES),
        "path_sources": list(PATH_SOURCE_VALUES),
    }


def default_extraction_prompt_template() -> str:
    page_types = "|".join(PAGE_TYPE_VALUES)
    statement_types = "|".join(STATEMENT_TYPE_VALUES)
    currencies = "|".join(CURRENCY_VALUES)
    scales = "|".join(str(value) for value in SCALE_VALUES)
    value_types = "|".join(VALUE_TYPE_VALUES)
    entity_types = "|".join(ENTITY_TYPE_VALUES)
    period_types = "|".join(PERIOD_TYPE_VALUES)
    path_sources = "|".join(PATH_SOURCE_VALUES)

    return dedent(
        f"""
        You are extracting financial-statement annotations from a single page image into the exact Pydantic schema `PageExtraction`.

        Return ONLY valid JSON.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Return a full document wrapper with exactly one page.
        Runtime will overwrite `images_dir` and `pages[0].image`, but you must still include them.

        Top-level schema (exact keys only):
        {{
          "images_dir": "<string or null>",
          "metadata": {{
            "language": "he|en|null",
            "reading_direction": "rtl|ltr|null",
            "company_name": "<string or null>",
            "company_id": "<string or null>",
            "report_year": "<integer or null>",
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
                  "currency": "{currencies}|null",
                  "scale": {scales}|null,
                  "date": "<YYYY|YYYY-MM|YYYY-MM-DD|null>",
                  "period_type": "{period_types}|null",
                  "period_start": "<YYYY-MM-DD|null>",
                  "period_end": "<YYYY-MM-DD|null>",
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
        9. `page_type` must be one of: `{page_types}`.
        10. `statement_type` must be one of: `{statement_types}` or null.
        11. `page_type="statements"` for balance sheet / income statement / cash flow / changes in equity / notes / board report / auditor report / statement of activities / other declaration pages.
        12. Non-statement structural pages use `statement_type=null`.
        13. `value_type` must be `amount`, `percent`, `ratio`, `count`, or null.
        14. If `value_type="percent"`, keep `%` inside `value` and set `currency` to null.
        15. `path_source` is only `observed`, `inferred`, or null.
        16. `note_num` must be a JSON integer or `null` only. Never emit a quoted number.
        17. If `note_num` is present, `note_flag` must be `true`.
        18. If `statement_type` is not `notes_to_financial_statements`, all facts must have `note_flag=false` and `note_num=null`.
        19. Use JSON `null` literal for missing optional values (never `"null"` string).
        20. Do not include any keys not listed above.

        Extraction rules:
        1. Extract all visible numeric/table facts, including negatives in parentheses and totals.
        2. If source value is exactly `-`, keep `value` as `"-"`. For `—` or `–`, still extract the fact and set `value` to empty string `""`.
        3. For non-percent values, normalize `value` to canonical numeric shape (`Num`, `Num.Num`, `(Num)`, `(Num.Num)`) or exact `"-"` when source is only `-`.
        4. For percent values, keep `%` in `value`.
        5. `bbox` must tightly cover the value text only, in pixel coordinates of the original image, as `[x, y, w, h]`.
        6. `comment_ref`: textual qualifier tied to the specific fact, else `null`.
        7. `note_ref`: use when the fact points or references another note; else `null`.
        8. `note_name`: note title/name only. Do not include generic prefixes such as `Note`, `note`, `באור`, or `ביאור`.
        9. `currency`: infer from page/header context when clear, else `null`.
        10. `scale`: `1` unless page/header indicates thousands or millions.
        11. `value_type`: use `amount` unless the fact is clearly a percent, ratio, or count.
        12. Preserve `date` when visible. Only populate `period_type`, `period_start`, and `period_end` when explicitly known from the page.
        13. `period_type="instant"` means the fact represents a value at a single point in time.
        14. `period_type="duration"` means the fact represents a value over a period between two dates.
        15. Use `path_source="observed"` when path labels are directly visible. Use `path_source="inferred"` only when hierarchy is reconstructed from layout/context.
        16. Order `facts` top-to-bottom; within each row use right-to-left for Hebrew/RTL pages and left-to-right for English/LTR pages (fallback RTL if uncertain).
        17. Output UTF-8 Hebrew directly (do not escape to unicode sequences).
        18. Do not emit empty-value facts unless the source explicitly shows a placeholder dash (`-`, `—`, `–`) for that cell.

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
    statement_types = "|".join(STATEMENT_TYPE_VALUES)
    period_types = "|".join(PERIOD_TYPE_VALUES)
    path_sources = "|".join(PATH_SOURCE_VALUES)

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

        Requested fact fields:
        {{{{FACT_FIELDS}}}}

        Requested meta fields:
        {{{{META_FIELDS}}}}

        Input snapshot JSON:
        {{{{REQUEST_JSON}}}}

        Output schema (exact keys only):
        {{
          "meta_updates": {{
            "statement_type": "{statement_types}|null"
          }},
          "fact_updates": [
            {{
              "fact_num": <integer >= 1>,
              "updates": {{
                "period_type": "{period_types}|null",
                "period_start": "<YYYY-MM-DD|null>",
                "period_end": "<YYYY-MM-DD|null>",
                "path_source": "{path_sources}|null"
              }}
            }}
          ]
        }}

        Rules:
        1. Return patch-only JSON. Never return full page annotations.
        2. Update only requested fields. Do not add non-requested fields.
        3. `fact_num` must match the provided snapshot identity.
        4. Keep `fact_updates` focused and minimal. Include only facts that need updates.
        5. Use JSON null (not string "null") for unknowns.
        6. If no confident update for a requested field, omit that field from `updates`.
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
    "PROMPT_TOP_LEVEL_KEYS",
    "REQUIRED_PROMPT_CANONICAL_KEYS",
    "SCALE_VALUES",
    "STATEMENT_TYPE_VALUES",
    "VALUE_TYPE_VALUES",
    "default_gemini_fill_prompt_template",
    "default_extraction_prompt_template",
    "schema_snapshot",
]
