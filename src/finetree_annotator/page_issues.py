from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Optional, Sequence

from .annotation_core import PageState, normalize_fact_data
from .date_normalization import normalize_date
from .equation_integrity import audit_and_rebuild_financial_facts
from .fact_normalization import normalize_note_num
from .schemas import PageType, StatementType, split_legacy_page_type

PageIssueSeverity = str

_PERCENT_VALUE_RE = re.compile(r"^-?\d+(?:\.\d+)?\s*%$")
_PATH_NOTE_WORD_RE = re.compile(r"\bnotes?\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PageIssue:
    severity: PageIssueSeverity
    code: str
    message: str
    page_image: str
    fact_index: Optional[int] = None
    field_name: Optional[str] = None


@dataclass(frozen=True)
class PageIssueSummary:
    page_image: str
    reg_flag_count: int = 0
    warning_count: int = 0
    issues: tuple[PageIssue, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DocumentIssueSummary:
    reg_flag_count: int = 0
    warning_count: int = 0
    pages_with_reg_flags: int = 0
    pages_with_warnings: int = 0
    page_summaries: dict[str, PageIssueSummary] = field(default_factory=dict)


def _non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def _append_issue(issue_map: dict[str, list[PageIssue]], issue: PageIssue) -> None:
    issue_map.setdefault(issue.page_image, []).append(issue)


def _summary_from_issues(page_image: str, issues: list[PageIssue]) -> PageIssueSummary:
    reg_flag_count = sum(1 for issue in issues if issue.severity == "reg_flag")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return PageIssueSummary(
        page_image=page_image,
        reg_flag_count=reg_flag_count,
        warning_count=warning_count,
        issues=tuple(issues),
    )


def _mixed_value_warning_issues(
    *,
    page_image: str,
    facts: list[dict[str, Any]],
    key: str,
    code: str,
    label: str,
) -> list[PageIssue]:
    by_value: dict[str, list[int]] = {}
    for fact_index, fact in enumerate(facts):
        text = _non_empty_text(fact.get(key))
        if not text:
            continue
        by_value.setdefault(text, []).append(fact_index)

    if len(by_value) <= 1:
        return []

    ordered_values = sorted(by_value.items(), key=lambda item: (-len(item[1]), item[0]))
    distinct_values_text = ", ".join(value for value, _ in ordered_values)
    dominant_value, dominant_indices = ordered_values[0]
    has_clear_majority = len(ordered_values) > 1 and len(dominant_indices) > len(ordered_values[1][1])

    issues: list[PageIssue] = []
    if has_clear_majority:
        for value, indices in ordered_values[1:]:
            for fact_index in indices:
                issues.append(
                    PageIssue(
                        severity="warning",
                        code=code,
                        message=(
                            f"Fact #{fact_index + 1} uses {label} '{value}', "
                            f"while most facts use '{dominant_value}'. "
                            f"Page values: {distinct_values_text}."
                        ),
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name=key,
                    )
                )
        return issues

    for value, indices in ordered_values:
        for fact_index in indices:
            issues.append(
                PageIssue(
                    severity="warning",
                    code=code,
                    message=(
                        f"Fact #{fact_index + 1} uses {label} '{value}' on a page with mixed "
                        f"{label} values: {distinct_values_text}."
                    ),
                    page_image=page_image,
                    fact_index=fact_index,
                    field_name=key,
                )
            )
    return issues


def _value_looks_percent(text: str) -> bool:
    return bool(_PERCENT_VALUE_RE.match(text.replace(" ", "")))


def _entity_key(text: str) -> str:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[\W_]+", "", lowered)


def _entity_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _entity_key(left), _entity_key(right)).ratio()


def _path_level_has_note_marker(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return bool(_PATH_NOTE_WORD_RE.search(lowered)) or "באור" in stripped or "ביאור" in stripped


def validate_page_issues(page_image: str, state: PageState) -> PageIssueSummary:
    issues_by_page = _validate_page_issue_lists([(page_image, state)])
    return issues_by_page.get(page_image, PageIssueSummary(page_image=page_image))


def _validate_page_issue_lists(page_states: Sequence[tuple[str, PageState]]) -> dict[str, PageIssueSummary]:
    issue_map: dict[str, list[PageIssue]] = {}
    for page_image, state in page_states:
        issues_by_page = issue_map.setdefault(page_image, [])
        records = list(state.facts or [])
        facts = [normalize_fact_data(record.fact) for record in records]
        page_meta = state.meta or {}
        page_type = _non_empty_text(page_meta.get("page_type")) or PageType.other.value
        statement_type = _non_empty_text(page_meta.get("statement_type"))
        if not statement_type:
            legacy_page_type, legacy_statement_type = split_legacy_page_type(page_meta.get("type"))
            if not _non_empty_text(page_meta.get("page_type")):
                page_type = legacy_page_type
            if legacy_statement_type is not None:
                statement_type = legacy_statement_type

        if facts and statement_type == StatementType.notes_to_financial_statements.value and not any(
            bool(fact.get("note_flag")) for fact in facts
        ):
            issues_by_page.append(
                PageIssue(
                    severity="warning",
                    code="notes_page_missing_note_flag",
                    message="Statement type is notes_to_financial_statements, but none of the facts are marked as note facts.",
                    page_image=page_image,
                    field_name="statement_type",
                )
            )

        if statement_type != StatementType.notes_to_financial_statements.value:
            for fact_index, fact in enumerate(facts):
                if bool(fact.get("note_flag")):
                    issues_by_page.append(
                        PageIssue(
                            severity="reg_flag",
                            code="note_flag_on_non_notes_page",
                            message=f"Fact #{fact_index + 1} is marked as note, but statement_type is {statement_type or 'null'}.",
                            page_image=page_image,
                            fact_index=fact_index,
                            field_name="note_flag",
                        )
                    )

        for fact_index, (record, fact) in enumerate(zip(records, facts)):
            value = _non_empty_text(fact.get("value"))
            raw_note_num = None
            if isinstance(record.fact, dict):
                raw_note_num = record.fact.get("note_num", record.fact.get("note", record.fact.get("beur_num")))
            note_num = fact.get("note_num")
            note_ref = _non_empty_text(fact.get("note_ref"))
            date_text = _non_empty_text(fact.get("date"))
            note_flag = bool(fact.get("note_flag"))
            note_name = _non_empty_text(fact.get("note_name"))
            value_type = _non_empty_text(fact.get("value_type"))
            currency = _non_empty_text(fact.get("currency"))
            scale = fact.get("scale")
            _normalized_note_num, note_num_warnings = normalize_note_num(raw_note_num)

            if not value:
                issues_by_page.append(
                    PageIssue(
                        severity="reg_flag",
                        code="fact_missing_value",
                        message=f"Fact #{fact_index + 1} has no value.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="value",
                    )
                )
            if note_num_warnings:
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="nonnumeric_note_num",
                        message=f"Fact #{fact_index + 1} has a note_num containing non-digit characters.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="note_num",
                    )
                )
            if note_flag and note_num is None and not note_num_warnings:
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="note_flag_missing_note_num",
                        message=f"Fact #{fact_index + 1} is marked as note, but note_num is empty.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="note_num",
                    )
                )
            if raw_note_num not in ("", None) and not note_flag:
                issues_by_page.append(
                    PageIssue(
                        severity="reg_flag",
                        code="note_num_without_note_flag",
                        message=f"Fact #{fact_index + 1} has note_num, but note flag is false.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="note_flag",
                    )
                )
            if date_text:
                _, date_warnings = normalize_date(date_text)
                if date_warnings:
                    issues_by_page.append(
                        PageIssue(
                            severity="reg_flag",
                            code="invalid_date",
                            message=f"Fact #{fact_index + 1} has an invalid date '{date_text}'.",
                            page_image=page_image,
                            fact_index=fact_index,
                            field_name="date",
                        )
                    )
            if statement_type == StatementType.notes_to_financial_statements.value and note_ref:
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="note_ref_on_notes_page",
                        message=f"Fact #{fact_index + 1} has note_ref on a notes_to_financial_statements page.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="note_ref",
                    )
                )
            if value_type == "percent" and scale is not None:
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="scale_on_percent_fact",
                        message=f"Fact #{fact_index + 1} is a percent value, but scale is set to '{scale}'.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="scale",
                    )
                )
            if value_type == "percent" and currency:
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="percent_with_currency",
                        message=f"Fact #{fact_index + 1} is a percent value, but currency is set to '{currency}'.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="currency",
                    )
                )
            if value_type == "amount" and _value_looks_percent(value):
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="amount_type_percent_value",
                        message=f"Fact #{fact_index + 1} is typed as amount, but value looks like a percent ('{value}').",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="value_type",
                    )
                )
            raw_path = record.fact.get("path") if isinstance(record.fact, dict) else None
            if isinstance(raw_path, list) and any(not str(item).strip() for item in raw_path):
                issues_by_page.append(
                    PageIssue(
                        severity="reg_flag",
                        code="path_contains_empty_level",
                        message=f"Fact #{fact_index + 1} has a path with empty levels.",
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="path",
                    )
                )
            for path_level in fact.get("path") or []:
                path_text = _non_empty_text(path_level)
                if not _path_level_has_note_marker(path_text):
                    continue
                issues_by_page.append(
                    PageIssue(
                        severity="warning",
                        code="path_contains_note_marker",
                        message=(
                            f"Fact #{fact_index + 1} path contains note marker '{path_text}'. "
                            "Prefer keeping note semantics in note_flag/note_name/note_num and using a cleaner business path."
                        ),
                        page_image=page_image,
                        fact_index=fact_index,
                        field_name="path",
                    )
                )
                break
            normalized_note_name = note_name.casefold()
            if normalized_note_name:
                for path_level in fact.get("path") or []:
                    path_text = _non_empty_text(path_level)
                    if not path_text:
                        continue
                    if normalized_note_name not in path_text.casefold():
                        continue
                    issues_by_page.append(
                        PageIssue(
                            severity="warning",
                            code="path_overlaps_note_name",
                            message=(
                                f"Fact #{fact_index + 1} path level '{path_text}' contains note_name '{note_name}'. "
                                "Prefer keeping note semantics in note_name and using a cleaner business path."
                            ),
                            page_image=page_image,
                            fact_index=fact_index,
                            field_name="path",
                        )
                    )
                    break
        warning_specs = (
            ("currency", "mixed_currency", "currency"),
            ("scale", "mixed_scale", "scale"),
            ("value_type", "mixed_value_type", "value type"),
        )
        for key, code, label in warning_specs:
            issues_by_page.extend(
                _mixed_value_warning_issues(
                    page_image=page_image,
                    facts=facts,
                    key=key,
                    code=code,
                    label=label,
                )
            )

        fact_index_by_num: dict[int, int] = {}
        for fact_index, fact in enumerate(facts):
            fact_num = fact.get("fact_num")
            if isinstance(fact_num, int) and fact_num >= 1:
                fact_index_by_num[fact_num] = fact_index

        _rebuilt_facts, integrity_findings = audit_and_rebuild_financial_facts(
            facts,
            statement_type=statement_type or None,
            apply_repairs=False,
        )
        suppressed_codes = {
            "row_role_corrected",
            "equation_rebuilt",
            "fact_equation_rebuilt",
        }
        for finding in integrity_findings:
            code = str(finding.get("code") or "").strip()
            if not code:
                continue
            if code in suppressed_codes:
                continue
            severity = str(finding.get("severity") or "warning").strip().lower()
            message = str(finding.get("message") or "").strip() or code
            field_name = str(finding.get("field_name") or "").strip() or None
            fact_num_raw = finding.get("fact_num")
            fact_index = None
            if isinstance(fact_num_raw, int):
                fact_index = fact_index_by_num.get(fact_num_raw)
            issues_by_page.append(
                PageIssue(
                    severity="reg_flag" if severity == "reg_flag" else "warning",
                    code=code,
                    message=message,
                    page_image=page_image,
                    fact_index=fact_index,
                    field_name=field_name,
                )
            )

    return {page_image: _summary_from_issues(page_image, issues) for page_image, issues in issue_map.items()}


def validate_document_issues(page_states: Sequence[tuple[str, PageState]]) -> DocumentIssueSummary:
    ordered_pages = list(page_states)
    issue_map = {
        page_image: list(summary.issues)
        for page_image, summary in _validate_page_issue_lists(ordered_pages).items()
    }
    for page_image, _state in ordered_pages:
        issue_map.setdefault(page_image, [])

    entity_rows: list[tuple[int, str, str]] = []
    page_num_rows: list[tuple[int, str, str]] = []
    page_type_rows: list[tuple[int, str, str]] = []
    for index, (page_image, state) in enumerate(ordered_pages):
        meta = state.meta or {}
        entity_rows.append((index, page_image, _non_empty_text(meta.get("entity_name"))))
        page_num_rows.append((index, page_image, _non_empty_text(meta.get("page_num"))))
        page_type = _non_empty_text(meta.get("page_type")) or PageType.other.value
        if not _non_empty_text(meta.get("page_type")) and _non_empty_text(meta.get("type")):
            page_type, _statement_type = split_legacy_page_type(meta.get("type"))
        page_type_rows.append((index, page_image, page_type))

    non_empty_entities = [(idx, page_image, entity) for idx, page_image, entity in entity_rows if entity]
    entity_counts = Counter(entity for _idx, _page, entity in non_empty_entities)
    if entity_counts:
        dominant_entity, dominant_count = entity_counts.most_common(1)[0]
        majority_threshold = max(2, math.floor(len(ordered_pages) / 2) + 1)
        if dominant_count >= majority_threshold:
            for _idx, page_image, entity in entity_rows:
                if not entity:
                    _append_issue(
                        issue_map,
                        PageIssue(
                            severity="warning",
                            code="missing_entity_name_among_majority",
                            message=f"Entity name is empty on this page, while most pages use '{dominant_entity}'.",
                            page_image=page_image,
                            field_name="entity_name",
                        ),
                    )

        if len(entity_counts) > 1:
            exact_values = list(entity_counts.keys())
            dominant_exact = max(exact_values, key=lambda value: (entity_counts[value], value))
            for entity in exact_values:
                if entity == dominant_exact:
                    continue
                if _entity_similarity(entity, dominant_exact) >= 0.84:
                    for _idx, page_image, current_entity in entity_rows:
                        if current_entity == entity:
                            _append_issue(
                                issue_map,
                                PageIssue(
                                    severity="warning",
                                    code="entity_name_variant",
                                    message=(
                                        f"Entity name '{entity}' differs slightly from the dominant document value "
                                        f"'{dominant_exact}'."
                                    ),
                                    page_image=page_image,
                                    field_name="entity_name",
                                ),
                            )

    duplicate_page_nums = {
        page_num: [page_image for _idx, page_image, candidate in page_num_rows if candidate == page_num]
        for page_num, count in Counter(page_num for _idx, _page, page_num in page_num_rows if page_num).items()
        if count > 1
    }
    for page_num, page_images in duplicate_page_nums.items():
        for page_image in page_images:
            _append_issue(
                issue_map,
                PageIssue(
                    severity="warning",
                    code="duplicate_page_num",
                    message=f"Page number '{page_num}' appears on multiple pages in this document.",
                    page_image=page_image,
                    field_name="page_num",
                ),
            )

    for idx, page_image, page_num in page_num_rows:
        if page_num:
            continue
        prev_num = page_num_rows[idx - 1][2] if idx > 0 else ""
        next_num = page_num_rows[idx + 1][2] if idx + 1 < len(page_num_rows) else ""
        if prev_num and next_num:
            _append_issue(
                issue_map,
                PageIssue(
                    severity="warning",
                    code="missing_page_num_between_numbered_pages",
                    message=f"Page number is missing here, but nearby pages use '{prev_num}' and '{next_num}'.",
                    page_image=page_image,
                    field_name="page_num",
                ),
            )

    previous_numeric: Optional[tuple[str, int]] = None
    for _idx, page_image, page_num in page_num_rows:
        if not page_num or not page_num.isdigit():
            continue
        current_numeric = int(page_num)
        if previous_numeric is not None:
            previous_page_num, previous_value = previous_numeric
            if current_numeric < previous_value:
                _append_issue(
                    issue_map,
                    PageIssue(
                        severity="warning",
                        code="page_num_backwards",
                        message=f"Page number goes backward from {previous_value} to {current_numeric}.",
                        page_image=page_image,
                        field_name="page_num",
                    ),
                )
            elif current_numeric > (previous_value + 1):
                _append_issue(
                    issue_map,
                    PageIssue(
                        severity="warning",
                        code="page_num_skip",
                        message=f"Page number jumps from {previous_value} to {current_numeric}.",
                        page_image=page_image,
                        field_name="page_num",
                    ),
                )
        previous_numeric = (page_image, current_numeric)

    run_start: Optional[int] = None
    for idx, (_page_image, page_type) in enumerate((page_image, page_type) for _i, page_image, page_type in page_type_rows):
        if page_type == PageType.other.value:
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None:
            run_end = idx - 1
            prev_idx = run_start - 1
            next_idx = idx
            if prev_idx >= 0 and next_idx < len(page_type_rows):
                prev_type = page_type_rows[prev_idx][2]
                next_type = page_type_rows[next_idx][2]
                if prev_type == next_type and prev_type != PageType.other.value:
                    for middle_idx in range(run_start, run_end + 1):
                        middle_page = page_type_rows[middle_idx][1]
                        _append_issue(
                            issue_map,
                            PageIssue(
                                severity="warning",
                                code="other_page_between_same_type_pages",
                                message=f"Page is typed as other, but it sits between {prev_type} pages.",
                                page_image=middle_page,
                                field_name="page_type",
                            ),
                        )
            run_start = None

    page_summaries = {
        page_image: _summary_from_issues(page_image, issues)
        for page_image, issues in issue_map.items()
    }
    return aggregate_document_issues(page_summaries)


def aggregate_document_issues(page_summaries: Mapping[str, PageIssueSummary]) -> DocumentIssueSummary:
    summaries = dict(page_summaries)
    reg_flag_count = sum(summary.reg_flag_count for summary in summaries.values())
    warning_count = sum(summary.warning_count for summary in summaries.values())
    pages_with_reg_flags = sum(1 for summary in summaries.values() if summary.reg_flag_count > 0)
    pages_with_warnings = sum(1 for summary in summaries.values() if summary.warning_count > 0)
    return DocumentIssueSummary(
        reg_flag_count=reg_flag_count,
        warning_count=warning_count,
        pages_with_reg_flags=pages_with_reg_flags,
        pages_with_warnings=pages_with_warnings,
        page_summaries=summaries,
    )


__all__ = [
    "DocumentIssueSummary",
    "PageIssue",
    "PageIssueSummary",
    "aggregate_document_issues",
    "validate_document_issues",
    "validate_page_issues",
]
