from __future__ import annotations

from textwrap import dedent
from typing import Any

from .schemas import Currency, ExtractedFact, PageExtraction, PageMeta, PageType, Scale, ValueType

PROMPT_TOP_LEVEL_KEYS: tuple[str, ...] = tuple(PageExtraction.model_fields.keys())
PAGE_META_KEYS: tuple[str, ...] = tuple(PageMeta.model_fields.keys())
EXTRACTED_FACT_KEYS: tuple[str, ...] = tuple(ExtractedFact.model_fields.keys())
CANONICAL_FACT_KEYS: tuple[str, ...] = tuple(key for key in EXTRACTED_FACT_KEYS if key != "bbox")
REQUIRED_PROMPT_CANONICAL_KEYS: tuple[str, ...] = ("ref_comment", "note_flag", "note_name", "note_num", "ref_note")
LEGACY_FACT_KEYS: tuple[str, ...] = (
    "is_beur",
    "beur_num",
    "refference",
    "reference",
    "ref",
    "beur",
    "beur_number",
    "footnote",
)
PAGE_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in PageType)
VALUE_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in ValueType)
CURRENCY_VALUES: tuple[str, ...] = tuple(item.value for item in Currency)
SCALE_VALUES: tuple[int, ...] = tuple(int(item.value) for item in Scale)


def schema_snapshot() -> dict[str, Any]:
    return {
        "page_extraction_keys": list(PROMPT_TOP_LEVEL_KEYS),
        "page_meta_keys": list(PAGE_META_KEYS),
        "extracted_fact_keys": list(EXTRACTED_FACT_KEYS),
        "canonical_fact_keys": list(CANONICAL_FACT_KEYS),
        "required_prompt_fact_keys": list(REQUIRED_PROMPT_CANONICAL_KEYS),
        "legacy_fact_keys": list(LEGACY_FACT_KEYS),
        "page_types": list(PAGE_TYPE_VALUES),
        "value_types": list(VALUE_TYPE_VALUES),
        "currencies": list(CURRENCY_VALUES),
        "scales": list(SCALE_VALUES),
    }


def default_extraction_prompt_template() -> str:
    page_types = "|".join(PAGE_TYPE_VALUES)
    currencies = "|".join(CURRENCY_VALUES)
    scales = "|".join(str(value) for value in SCALE_VALUES)
    value_types = "|".join(VALUE_TYPE_VALUES)

    return dedent(
        f"""
        You are extracting financial-statement annotations from a single page image into the exact Pydantic schema `PageExtraction`.

        Return ONLY valid JSON.
        Do NOT return markdown, code fences, comments, prose, or extra keys.

        Top-level schema (exact keys only):
        {{
          "meta": {{
            "entity_name": "<string or null>",
            "page_num": "<string or null>",
            "type": "{page_types}",
            "title": "<string or null>"
          }},
          "facts": [
            {{
              "bbox": {{ "x": <number>, "y": <number>, "w": <number>, "h": <number> }},
              "value": "<string>",
              "ref_comment": "<string or null>",
              "note_flag": <true|false>,
              "note_name": "<string or null>",
              "note_num": "<integer or null>",
              "ref_note": "<string or null>",
              "date": "<string or null>",
              "path": ["<string>", "..."],
              "currency": "{currencies}|null",
              "scale": {scales}|null,
              "value_type": "{value_types}|null"
            }}
          ]
        }}

        Type/validation rules (must match schema):
        1. `meta` and `facts` are required.
        2. All listed `meta` keys are required: `entity_name`, `page_num`, `type`, `title`.
        3. All listed fact keys are required in every fact object, including `bbox`; nullable fields must still be present with JSON null.
        4. `ref_note` is required and nullable. Use JSON `null` when missing (never `""`).
        5. `path` is a required list of strings (use `[]` when unknown, never `null`).
        6. `note_flag` must be boolean (never null and never `"true"`/`"false"` strings).
        7. `date` must be `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` when provided.
        8. `value` rules:
           - If the value is a percentage, include `%` (for example `12%`, `12.5%`).
           - If not a percentage, output only canonical numeric text: `Num`, `Num.Num`, `(Num)`, `(Num.Num)`, or exact `"-"` when source is only `-`.
           - For non-`%` values remove commas, spaces, currency signs/codes, and use parentheses for negatives (never leading `-`).
        9. Use JSON `null` literal for missing optional values (never `"null"` string).
        10. Do not include any keys not listed above.
        11. `note_name` is required and nullable. Use it for the note title/name only, without generic prefixes like `Note`, `note`, `באור`, or `ביאור`.
        12. `note_num` must be a JSON integer or `null` only. Never emit a quoted number.
        13. If `note_num` is present, `note_flag` must be `true`.
        14. If `meta.type` is not `notes`, all facts must have `note_flag=false` and `note_num=null`.
        15. `path` entries must be non-empty strings only; never output `""` inside `path`.

        Extraction rules:
        1. Extract all visible numeric/table facts, including negatives in parentheses and totals.
        2. If source value is exactly `-`, keep `value` as `"-"`. For `—` or `–`, still extract the fact and set `value` to empty string `""`.
        3. For non-`%` values, normalize `value` to canonical numeric shape (`Num`, `Num.Num`, `(Num)`, `(Num.Num)`) or exact `"-"` when source is only `-`.
        4. For `%` values, keep `%` in `value`.
        5. `bbox` must tightly cover the value text only, in pixel coordinates of the original image.
        6. `ref_comment`: textual qualifier tied to the specific fact (for example `*without debt insurance`), else `null`.
        7. `note_flag`: `true` when the fact is itself inside a note section, else `false`.
        8. `note_name`: note title/name only (for example `רכוש אחר`). Do not include generic prefixes such as `Note`, `note`, `באור`, or `ביאור` here.
        9. `note_num`: integer note number for facts that belong to note content; else `null`.
        10. `ref_note`: use when this fact points/references another note; else `null`. If the visible note marker is not a plain integer (for example `2ה׳`, `A1`), use `ref_note`, not `note_num`.
        11. `currency`: infer from page/header context (for example "שקלים חדשים" -> "ILS"), else `null`.
        12. `scale`: `1` unless page/header indicates thousands or millions. For percentage facts, prefer `scale=null` unless the page truly expresses scaled percentages.
        13. `value_type`: `"amount"` unless the fact is a percentage (`"%"`). If `value_type` is `"%"`, set `currency` to `null`.
        14. `date`: output only `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. Convert dotted dates (`D.M.YYYY` / `DD.MM.YYYY`) to `YYYY-MM-DD` using day-first parsing.
        15. Order `facts` top-to-bottom; within each row use right-to-left for Hebrew/RTL pages and left-to-right for English/LTR pages (fallback RTL if uncertain).
        16. Output UTF-8 Hebrew directly (do not escape to unicode sequences).
        17. Do not emit empty-value facts unless the source explicitly shows a placeholder dash (`-`, `—`, `–`) for that cell.
        18. On `notes` pages, prefer `note_flag=true` for note-body facts. On non-`notes` pages, use `note_flag=false` unless the page is clearly mislabeled.

        Path hierarchy rule:
        1. Build `path` with horizontal-axis hierarchy first (row-side structure: group -> subgroup -> line item).
        2. Then append vertical-axis hierarchy only when it adds business semantics not already represented by dedicated fields.
        3. Do not encode period/currency/scale/value-format in `path` when already represented by `date`/`currency`/`scale`/`value_type`.
        4. Keep labels exactly as shown. Do not invent levels.
        5. Do not include generic note markers like `Note`, `note`, `באור`, or `ביאור` inside `path`. Put note semantics in `note_flag`, `note_num`, `note_name`, and `ref_note` instead.
        """
    ).strip()


__all__ = [
    "CANONICAL_FACT_KEYS",
    "CURRENCY_VALUES",
    "EXTRACTED_FACT_KEYS",
    "LEGACY_FACT_KEYS",
    "PAGE_META_KEYS",
    "PAGE_TYPE_VALUES",
    "PROMPT_TOP_LEVEL_KEYS",
    "REQUIRED_PROMPT_CANONICAL_KEYS",
    "SCALE_VALUES",
    "VALUE_TYPE_VALUES",
    "default_extraction_prompt_template",
    "schema_snapshot",
]
