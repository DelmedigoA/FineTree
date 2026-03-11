from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .fact_normalization import normalize_fact_payload

_EQUATION_NUMERIC_VALUE_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
_FACT_EQUATION_RE = re.compile(r"^\s*[+-]?\s*f\d+(?:\s*[+-]\s*f\d+)*\s*$", flags=re.IGNORECASE)
_FACT_EQUATION_TOKEN_RE = re.compile(r"([+-]?)\s*f(\d+)", flags=re.IGNORECASE)
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


def _normalize_row_role(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"detail", "total"}:
        return text
    return None


def _natural_sign_multiplier(value: Any) -> int:
    return -1 if _normalize_natural_sign(value) == "negative" else 1


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


def _parse_fact_equation_children(value: Any) -> list[dict[str, Any]] | None:
    text = str(value or "").strip()
    if not text:
        return []
    if not _FACT_EQUATION_RE.fullmatch(text):
        return None
    children: list[dict[str, Any]] = []
    for sign_text, fact_num_text in _FACT_EQUATION_TOKEN_RE.findall(text):
        operator = "-" if sign_text.strip() == "-" else "+"
        children.append({"fact_num": int(fact_num_text), "operator": operator})
    return children


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


def _remap_equation_children_references(
    equation_children: Any,
    fact_num_remap: Mapping[int, int],
    *,
    ambiguous_old_nums: set[int] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(equation_children, list):
        return None
    ambiguous = set(ambiguous_old_nums or set())
    out: list[dict[str, Any]] = []
    for entry in equation_children:
        if not isinstance(entry, Mapping):
            continue
        raw_fact_num = entry.get("fact_num")
        if not isinstance(raw_fact_num, int):
            continue
        next_fact_num = raw_fact_num if raw_fact_num in ambiguous else fact_num_remap.get(raw_fact_num, raw_fact_num)
        operator = str(entry.get("operator") or "").strip()
        if operator not in {"+", "-"}:
            continue
        out.append({"fact_num": next_fact_num, "operator": operator})
    return out or None


def _equation_variant_signature(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    equation = str(entry.get("equation") or "").strip()
    fact_equation = str(entry.get("fact_equation") or "").strip()
    children = tuple(
        (int(child.get("fact_num")), str(child.get("operator") or "+"))
        for child in (entry.get("equation_children") or [])
        if isinstance(child, Mapping) and isinstance(child.get("fact_num"), int)
    )
    return equation, fact_equation, children


def _remap_equation_variants(
    equations: Any,
    fact_num_remap: Mapping[int, int],
    *,
    ambiguous_old_nums: set[int] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(equations, list):
        return None
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in equations:
        if not isinstance(entry, Mapping):
            continue
        equation = str(entry.get("equation") or "").strip()
        if not equation:
            continue
        remapped_fact_equation = remap_fact_equation_references(
            entry.get("fact_equation"),
            fact_num_remap,
            ambiguous_old_nums=ambiguous_old_nums,
        )
        remapped_children = _remap_equation_children_references(
            entry.get("equation_children"),
            fact_num_remap,
            ambiguous_old_nums=ambiguous_old_nums,
        )
        normalized_entry = {
            "equation": equation,
            "fact_equation": remapped_fact_equation,
            "equation_children": remapped_children,
        }
        signature = _equation_variant_signature(normalized_entry)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(normalized_entry)
    return out or None


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
        remapped_children = _remap_equation_children_references(
            updated.get("equation_children"),
            fact_num_remap,
            ambiguous_old_nums=ambiguous_old_nums,
        )
        if remapped_children is not None:
            updated["equation_children"] = remapped_children
        remapped_fact_equation = remap_fact_equation_references(
            updated.get("fact_equation"),
            fact_num_remap,
            ambiguous_old_nums=ambiguous_old_nums,
        )
        if remapped_fact_equation != updated.get("fact_equation"):
            updated["fact_equation"] = remapped_fact_equation
        remapped_equations = _remap_equation_variants(
            updated.get("equations"),
            fact_num_remap,
            ambiguous_old_nums=ambiguous_old_nums,
        )
        if remapped_equations is not None:
            updated["equations"] = remapped_equations
            active_variant = remapped_equations[0]
            updated["equation"] = active_variant.get("equation")
            updated["fact_equation"] = active_variant.get("fact_equation")
            updated["equation_children"] = active_variant.get("equation_children")
        resequenced.append(updated)

    rebuilt, _findings = audit_and_rebuild_financial_facts(resequenced, apply_repairs=True)
    return rebuilt


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
    return target_value


def _build_equation_from_children(
    equation_children: Sequence[Mapping[str, Any]],
    facts_by_num: Mapping[int, Mapping[str, Any]],
) -> tuple[str | None, str | None, Decimal | None, list[int]]:
    numeric_terms: list[str] = []
    fact_terms: list[str] = []
    total = Decimal("0")
    invalid_refs: list[int] = []
    valid_count = 0

    for entry in equation_children:
        ref = entry.get("fact_num")
        operator = str(entry.get("operator") or "").strip()
        if not isinstance(ref, int) or operator not in {"+", "-"}:
            continue
        fact = facts_by_num.get(ref)
        if fact is None:
            invalid_refs.append(ref)
            continue
        parsed, display = _parse_fact_value_for_equation(fact.get("value"))
        if parsed is None or display is None:
            invalid_refs.append(ref)
            continue

        contribution = parsed * Decimal(-1 if operator == "-" else 1)
        rendered_sign = -1 if contribution < 0 else 1
        prefix = ""
        if valid_count > 0:
            prefix = "- " if rendered_sign < 0 else "+ "
        elif rendered_sign < 0:
            prefix = "- "
        numeric_terms.append(f"{prefix}{display}" if prefix else display)
        fact_terms.append(f"{prefix}f{ref}" if prefix else f"f{ref}")
        total += contribution
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


def _cycle_nodes(facts_by_num: Mapping[int, Mapping[str, Any]]) -> set[int]:
    children_by_num: dict[int, list[int]] = {}
    for fact_num, fact in facts_by_num.items():
        entries = fact.get("equation_children")
        if not isinstance(entries, list):
            continue
        children_by_num[fact_num] = [
            int(entry["fact_num"])
            for entry in entries
            if isinstance(entry, Mapping) and isinstance(entry.get("fact_num"), int)
        ]

    cycle_nodes: set[int] = set()
    visiting: set[int] = set()
    visited: set[int] = set()

    def _dfs(node: int, stack: list[int]) -> None:
        if node in visiting:
            if node in stack:
                cycle_nodes.update(stack[stack.index(node):])
            else:
                cycle_nodes.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for child in children_by_num.get(node, []):
            if child in children_by_num:
                _dfs(child, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for fact_num in children_by_num:
        _dfs(fact_num, [])
    return cycle_nodes


def audit_and_rebuild_financial_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    statement_type: str | None = None,
    apply_repairs: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _ = statement_type
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
        equation_children = fact.get("equation_children")
        if equation_children is None and saved_fact_equation:
            parsed_children = _parse_fact_equation_children(saved_fact_equation)
            if parsed_children is None:
                findings.append(
                    {
                        "code": "invalid_fact_equation_syntax",
                        "severity": "reg_flag",
                        "fact_num": fact_num,
                        "field_name": "fact_equation",
                        "message": "fact_equation has invalid syntax; expected terms like 'f1 - f2 + f3'.",
                    }
                )
            elif parsed_children:
                if apply_repairs:
                    fact["equation_children"] = parsed_children
                equation_children = parsed_children
            elif saved_equation:
                findings.append(
                    {
                        "code": "equation_missing_fact_equation",
                        "severity": "warning",
                        "fact_num": fact_num,
                        "field_name": "fact_equation",
                        "message": "equation exists without fact_equation references.",
                    }
                )

        row_role = _normalize_row_role(fact.get("row_role")) or "detail"
        if row_role == "detail" and fact.get("equation_children") is not None:
            findings.append(
                {
                    "code": "detail_has_equation_children",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation_children",
                    "message": "detail rows must not contain equation_children.",
                }
            )
            if apply_repairs:
                fact["equation_children"] = None
        if row_role == "total" and not fact.get("equation_children"):
            findings.append(
                {
                    "code": "total_missing_equation_children",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation_children",
                    "message": "total rows must contain at least one equation_children entry.",
                }
            )

    facts_by_num = {
        int(fact_num): fact
        for fact in normalized_facts
        for fact_num in [fact.get("fact_num")]
        if isinstance(fact_num, int) and fact_num >= 1
    }
    cycle_nodes = _cycle_nodes(facts_by_num)

    for fact in normalized_facts:
        fact_num = fact.get("fact_num") if isinstance(fact.get("fact_num"), int) else None
        equation_children = fact.get("equation_children")
        if not isinstance(equation_children, list):
            continue

        seen_refs: set[int] = set()
        duplicate_refs: list[int] = []
        missing_refs: list[int] = []
        self_refs: list[int] = []
        normalized_children: list[dict[str, Any]] = []
        for entry in equation_children:
            if not isinstance(entry, Mapping):
                continue
            ref = entry.get("fact_num")
            operator = str(entry.get("operator") or "").strip()
            if not isinstance(ref, int) or operator not in {"+", "-"}:
                continue
            normalized_children.append({"fact_num": ref, "operator": operator})
            if ref == fact_num:
                self_refs.append(ref)
            if ref in seen_refs:
                duplicate_refs.append(ref)
            else:
                seen_refs.add(ref)
            if ref not in facts_by_num:
                missing_refs.append(ref)

        if normalized_children != equation_children and apply_repairs:
            fact["equation_children"] = normalized_children or None
            equation_children = fact.get("equation_children")

        if duplicate_refs:
            findings.append(
                {
                    "code": "duplicate_equation_child_reference",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation_children",
                    "message": f"equation_children includes duplicate child references: {sorted(set(duplicate_refs))}.",
                }
            )
        if self_refs:
            findings.append(
                {
                    "code": "equation_children_self_reference",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation_children",
                    "message": "equation_children must not reference the parent fact itself.",
                }
            )
        if missing_refs:
            findings.append(
                {
                    "code": "fact_equation_missing_reference",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation_children",
                    "message": f"equation_children references missing facts: {sorted(set(missing_refs))}.",
                }
            )
        if fact_num in cycle_nodes:
            findings.append(
                {
                    "code": "equation_graph_cycle",
                    "severity": "reg_flag",
                    "fact_num": fact_num,
                    "field_name": "equation_children",
                    "message": "equation_children graph must be acyclic.",
                }
            )

        rebuilt_equation, rebuilt_fact_equation, rebuilt_total, invalid_refs = _build_equation_from_children(
            normalized_children,
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

        unresolved_rebuild = bool(missing_refs or invalid_refs or duplicate_refs or self_refs or fact_num in cycle_nodes)
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

        saved_equation = str(fact.get("equation") or "").strip()
        saved_fact_equation = str(fact.get("fact_equation") or "").strip()
        if saved_fact_equation != rebuilt_fact_equation or _fact_equation_ref_set(saved_fact_equation) != {
            child["fact_num"] for child in normalized_children
        }:
            findings.append(
                {
                    "code": "fact_equation_rebuilt",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "fact_equation",
                    "message": "fact_equation references/order were rebuilt from equation_children.",
                }
            )
        if saved_equation != rebuilt_equation:
            findings.append(
                {
                    "code": "equation_rebuilt",
                    "severity": "warning",
                    "fact_num": fact_num,
                    "field_name": "equation",
                    "message": "equation text was rebuilt from equation_children.",
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
