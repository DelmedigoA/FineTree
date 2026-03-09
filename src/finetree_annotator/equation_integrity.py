from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .fact_normalization import normalize_fact_payload

_EQUATION_NUMERIC_VALUE_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
_FACT_EQUATION_RE = re.compile(r"^\s*[+-]?\s*f\d+(?:\s*[+-]\s*f\d+)*\s*$", flags=re.IGNORECASE)
_FACT_EQUATION_REF_RE = re.compile(r"[+-]?\s*f(\d+)", flags=re.IGNORECASE)
_FACT_EQUATION_FACTNUM_RE = re.compile(r"f(\d+)", flags=re.IGNORECASE)

_ROW_TOTAL_MARKERS: tuple[str, ...] = (
    'סה"כ',
    "סה״כ",
    "סהכ",
    "סך",
    "סך הכל",
    "total",
    "subtotal",
    "net subtotal",
    "net total",
    "net",
)
_AGGREGATION_SUBTRACTIVE_MARKERS: tuple[str, ...] = (
    "בניכוי",
    "פחות",
    "less",
    "minus",
    "net of",
    "contra",
    "accumulated depreciation",
    "פחת שנצבר",
)


def _format_decimal_plain(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    return text


def _normalize_fact(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized, _warnings = normalize_fact_payload(dict(data), include_bbox=False)
    return normalized


def _normalize_text_for_role_inference(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = (
        text.replace("״", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("׳", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    return re.sub(r"\s+", " ", text)


def _normalize_natural_sign(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "negative":
        return "negative"
    if text == "positive":
        return "positive"
    return None


def _normalize_aggregation_role(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"additive", "subtractive"}:
        return text
    if text in {"total", "unknown"}:
        return "additive"
    return None


def _normalize_row_role(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"detail", "total"}:
        return text
    return None


def _natural_sign_multiplier(value: Any) -> int:
    return -1 if _normalize_natural_sign(value) == "negative" else 1


def _aggregation_role_multiplier(value: Any) -> int:
    return -1 if _normalize_aggregation_role(value) == "subtractive" else 1


def _contribution_multiplier(natural_sign: Any, aggregation_role: Any) -> int:
    return _natural_sign_multiplier(natural_sign) * _aggregation_role_multiplier(aggregation_role)


def _parse_fact_value_for_equation(value: Any) -> tuple[Decimal | None, str | None]:
    raw = str(value or "").strip()
    if not raw:
        return None, None
    if raw == "-":
        return Decimal("0"), "0"

    negative = False
    text = raw
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.startswith("+"):
        text = text[1:].strip()
    elif text.startswith("-"):
        negative = True
        text = text[1:].strip()
    if not _EQUATION_NUMERIC_VALUE_RE.fullmatch(text):
        return None, None
    try:
        parsed = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None, None
    return (-parsed if negative else parsed), text


def _parse_fact_equation_refs(value: Any) -> list[int] | None:
    text = str(value or "").strip()
    if not text:
        return []
    if not _FACT_EQUATION_RE.fullmatch(text):
        return None
    refs = [int(match) for match in _FACT_EQUATION_REF_RE.findall(text)]
    return refs


def _fact_equation_ref_set(value: Any) -> set[int]:
    text = str(value or "")
    return {int(match) for match in _FACT_EQUATION_FACTNUM_RE.findall(text)}


def remap_fact_equation_references(
    fact_equation: Any,
    fact_num_remap: Mapping[int, int],
    *,
    ambiguous_old_nums: set[int] | None = None,
) -> str | None:
    text = str(fact_equation or "").strip()
    if not text:
        return None
    if not _FACT_EQUATION_RE.fullmatch(text):
        return text
    ambiguous = set(ambiguous_old_nums or set())

    def _replace(match: re.Match[str]) -> str:
        old_num = int(match.group(1))
        if old_num in ambiguous:
            return match.group(0)
        new_num = fact_num_remap.get(old_num)
        if new_num is None:
            return match.group(0)
        return f"f{new_num}"

    return _FACT_EQUATION_FACTNUM_RE.sub(_replace, text)


def resequence_fact_numbers_and_remap_fact_equations(
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_facts = [_normalize_fact(fact) for fact in facts]

    fact_num_remap: dict[int, int] = {}
    ambiguous_old_nums: set[int] = set()
    for next_fact_num, fact in enumerate(normalized_facts, start=1):
        old_fact_num = fact.get("fact_num")
        if not isinstance(old_fact_num, int) or old_fact_num < 1:
            continue
        prior = fact_num_remap.get(old_fact_num)
        if prior is None:
            fact_num_remap[old_fact_num] = next_fact_num
        elif prior != next_fact_num:
            ambiguous_old_nums.add(old_fact_num)

    resequenced: list[dict[str, Any]] = []
    for next_fact_num, fact in enumerate(normalized_facts, start=1):
        updated = dict(fact)
        updated["fact_num"] = next_fact_num
        remapped_fact_equation = remap_fact_equation_references(
            updated.get("fact_equation"),
            fact_num_remap,
            ambiguous_old_nums=ambiguous_old_nums,
        )
        if remapped_fact_equation != updated.get("fact_equation"):
            updated["fact_equation"] = remapped_fact_equation
        resequenced.append(updated)
    return resequenced


def _inferred_aggregation_role(fact: Mapping[str, Any], *, statement_type: str | None) -> str:
    contexts: list[str] = []
    for key in ("comment_ref", "note_name", "note_ref"):
        normalized = _normalize_text_for_role_inference(fact.get(key))
        if normalized:
            contexts.append(normalized)
    for level in fact.get("path") or []:
        normalized = _normalize_text_for_role_inference(level)
        if normalized:
            contexts.append(normalized)

    for text in contexts:
        if any(marker in text for marker in _AGGREGATION_SUBTRACTIVE_MARKERS):
            return "subtractive"
    # Cash-flow detail defaults to additive; all other non-marker rows do as well.
    if str(statement_type or "").strip().lower() == "cash_flow_statement":
        return "additive"
    return "additive"


def _inferred_row_role(fact: Mapping[str, Any]) -> str:
    contexts: list[str] = []
    for key in ("comment_ref", "note_name", "note_ref"):
        normalized = _normalize_text_for_role_inference(fact.get(key))
        if normalized:
            contexts.append(normalized)
    for level in fact.get("path") or []:
        normalized = _normalize_text_for_role_inference(level)
        if normalized:
            contexts.append(normalized)
    for text in contexts:
        if any(marker in text for marker in _ROW_TOTAL_MARKERS):
            return "total"
    return "detail"


def _effective_target_value(fact: Mapping[str, Any]) -> Decimal | None:
    target_value, _display = _parse_fact_value_for_equation(fact.get("value"))
    if target_value is None:
        return None
    return target_value.copy_abs() * Decimal(
        _contribution_multiplier(fact.get("natural_sign"), fact.get("aggregation_role"))
    )


def _build_equation_from_refs(
    refs_in_display_order: Sequence[int],
    facts_by_num: Mapping[int, Mapping[str, Any]],
) -> tuple[str | None, str | None, Decimal | None, list[int]]:
    numeric_terms: list[str] = []
    fact_terms: list[str] = []
    total = Decimal("0")
    invalid_refs: list[int] = []
    valid_count = 0

    for ref in refs_in_display_order:
        fact = facts_by_num.get(ref)
        if fact is None:
            invalid_refs.append(ref)
            continue
        parsed, display = _parse_fact_value_for_equation(fact.get("value"))
        if parsed is None or display is None:
            invalid_refs.append(ref)
            continue
        magnitude = parsed.copy_abs()
        effective_value = magnitude * Decimal(
            _contribution_multiplier(fact.get("natural_sign"), fact.get("aggregation_role"))
        )
        prefix = ""
        if valid_count > 0:
            prefix = "- " if effective_value < 0 else "+ "
        elif effective_value < 0:
            prefix = "- "
        numeric_terms.append(f"{prefix}{display}" if prefix else display)
        fact_terms.append(f"{prefix}f{ref}" if prefix else f"f{ref}")
        total += effective_value
        valid_count += 1

    if valid_count == 0:
        return None, None, None, invalid_refs
    return " ".join(numeric_terms), " ".join(fact_terms), total, invalid_refs


def _enforce_period_integrity(
    fact: dict[str, Any],
    *,
    findings: list[dict[str, Any]],
    fact_num: int | None,
    apply_repairs: bool,
) -> None:
    period_type = str(fact.get("period_type") or "").strip().lower()
    period_start = str(fact.get("period_start") or "").strip()
    period_end = str(fact.get("period_end") or "").strip()
    date_value = str(fact.get("date") or "").strip()

    if period_type == "instant" and period_start:
        findings.append(
            {
                "code": "instant_period_start_not_null",
                "severity": "reg_flag",
                "fact_num": fact_num,
                "field_name": "period_start",
                "message": "period_type='instant' requires period_start to be null.",
            }
        )
        if apply_repairs:
            fact["period_start"] = None
            period_start = ""

    if period_type != "duration":
        return

    if not period_start or not period_end:
        findings.append(
            {
                "code": "duration_missing_period_bounds",
                "severity": "reg_flag",
                "fact_num": fact_num,
                "field_name": "period_end" if not period_end else "period_start",
                "message": "period_type='duration' requires both period_start and period_end.",
            }
        )
        return

    if period_start > period_end:
        findings.append(
            {
                "code": "duration_period_bounds_order_invalid",
                "severity": "reg_flag",
                "fact_num": fact_num,
                "field_name": "period_start",
                "message": "period_start must be less than or equal to period_end for duration facts.",
            }
        )

    if not date_value:
        return
    date_year = date_value[:4]
    start_year = period_start[:4]
    end_year = period_end[:4]
    if date_year and (date_year != start_year or date_year != end_year):
        findings.append(
            {
                "code": "duration_date_year_mismatch",
                "severity": "warning",
                "fact_num": fact_num,
                "field_name": "date",
                "message": "duration fact has date/year that does not match period_start/period_end year.",
            }
        )


def audit_and_rebuild_financial_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    statement_type: str | None = None,
    apply_repairs: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_facts = [_normalize_fact(fact) for fact in facts]
    findings: list[dict[str, Any]] = []
    if not normalized_facts:
        return normalized_facts, findings

    for fact in normalized_facts:
        fact_num = fact.get("fact_num") if isinstance(fact.get("fact_num"), int) else None
        inferred_row_role = _inferred_row_role(fact)
        current_row_role = _normalize_row_role(fact.get("row_role"))
        next_row_role = current_row_role if current_row_role in {"detail", "total"} else inferred_row_role
        if next_row_role != current_row_role:
            findings.append(
                {
                    "code": "row_role_corrected",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "row_role",
                    "message": f"row_role corrected from {current_row_role or 'null'} to {next_row_role}.",
                }
            )
            if apply_repairs:
                fact["row_role"] = next_row_role

        inferred_role = _inferred_aggregation_role(fact, statement_type=statement_type)
        current_role = _normalize_aggregation_role(fact.get("aggregation_role"))
        next_role = current_role
        # Keep explicit user-selected aggregation contribution; only infer when missing.
        if current_role is None:
            next_role = inferred_role if inferred_role in {"additive", "subtractive"} else "additive"
        if next_role != current_role:
            findings.append(
                {
                    "code": "aggregation_role_corrected",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "aggregation_role",
                    "message": f"aggregation_role corrected from {current_role or 'null'} to {next_role}.",
                }
            )
            if apply_repairs:
                fact["aggregation_role"] = next_role

        _enforce_period_integrity(
            fact,
            findings=findings,
            fact_num=fact_num,
            apply_repairs=apply_repairs,
        )

    facts_by_num = {
        int(fact_num): fact
        for fact in normalized_facts
        for fact_num in [fact.get("fact_num")]
        if isinstance(fact_num, int) and fact_num >= 1
    }

    for fact in normalized_facts:
        fact_num = fact.get("fact_num") if isinstance(fact.get("fact_num"), int) else None
        saved_equation = str(fact.get("equation") or "").strip()
        saved_fact_equation = str(fact.get("fact_equation") or "").strip()
        if not saved_equation and not saved_fact_equation:
            continue

        parsed_refs = _parse_fact_equation_refs(saved_fact_equation)
        if parsed_refs is None:
            findings.append(
                {
                    "code": "invalid_fact_equation_syntax",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "fact_equation",
                    "message": "fact_equation has invalid syntax; expected terms like 'f1 - f2 + f3'.",
                }
            )
            continue
        if not parsed_refs:
            findings.append(
                {
                    "code": "equation_missing_fact_equation",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "fact_equation",
                    "message": "equation exists without fact_equation references.",
                }
            )
            continue

        unique_refs: list[int] = []
        seen: set[int] = set()
        duplicate_refs: list[int] = []
        for ref in parsed_refs:
            if ref in seen:
                duplicate_refs.append(ref)
                continue
            seen.add(ref)
            unique_refs.append(ref)
        if duplicate_refs:
            findings.append(
                {
                    "code": "duplicate_fact_equation_reference",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "fact_equation",
                    "message": f"fact_equation had duplicate references: {sorted(set(duplicate_refs))}.",
                }
            )

        missing_refs = [ref for ref in unique_refs if ref not in facts_by_num]
        if missing_refs:
            findings.append(
                {
                    "code": "fact_equation_missing_reference",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "fact_equation",
                    "message": f"fact_equation references missing facts: {missing_refs}.",
                }
            )

        existing_refs = [ref for ref in unique_refs if ref in facts_by_num]
        ordered_refs = sorted(existing_refs)
        rebuilt_equation, rebuilt_fact_equation, rebuilt_total, invalid_refs = _build_equation_from_refs(
            ordered_refs,
            facts_by_num,
        )
        if invalid_refs:
            findings.append(
                {
                    "code": "fact_equation_invalid_child_value",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation",
                    "message": f"Referenced child facts are non-numeric for equation arithmetic: {sorted(set(invalid_refs))}.",
                }
            )

        unresolved_rebuild = bool(missing_refs or invalid_refs)
        if rebuilt_equation is None or rebuilt_fact_equation is None or rebuilt_total is None:
            unresolved_rebuild = True
        if unresolved_rebuild:
            findings.append(
                {
                    "code": "equation_rebuild_unresolved",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation",
                    "message": "Could not rebuild equation from referenced child facts.",
                }
            )
            continue

        if saved_fact_equation != rebuilt_fact_equation or _fact_equation_ref_set(saved_fact_equation) != set(ordered_refs):
            findings.append(
                {
                    "code": "fact_equation_rebuilt",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "fact_equation",
                    "message": "fact_equation references/order were rebuilt from existing child facts.",
                }
            )
        if saved_equation != rebuilt_equation:
            findings.append(
                {
                    "code": "equation_rebuilt",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "equation",
                    "message": "equation text was rebuilt from referenced child facts.",
                }
            )
        if apply_repairs:
            fact["equation"] = rebuilt_equation
            fact["fact_equation"] = rebuilt_fact_equation

        target_value = _effective_target_value(fact)
        if target_value is None:
            findings.append(
                {
                    "code": "equation_target_non_numeric",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "value",
                    "message": "Target fact value is non-numeric; arithmetic integrity cannot be checked.",
                }
            )
            continue
        if rebuilt_total != target_value:
            findings.append(
                {
                    "code": "equation_arithmetic_mismatch",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation",
                    "message": (
                        f"Subtotal mismatch: children sum to {_format_decimal_plain(rebuilt_total)}, "
                        f"target is {_format_decimal_plain(target_value)}."
                    ),
                }
            )

    return normalized_facts, findings


__all__ = [
    "audit_and_rebuild_financial_facts",
    "remap_fact_equation_references",
    "resequence_fact_numbers_and_remap_fact_equations",
]
