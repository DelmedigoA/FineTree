from __future__ import annotations

"""
Migration helper: add period_* fields for facts with legacy "date" and remove suggested_period_* fields.

Rules:
- Preserve "date" always.
- Add missing period_type/period_start/period_end as null.
- If period_* are missing, fill them using the date heuristics:
  - YYYY-MM-DD -> period_type="instant", period_start=null, period_end=YYYY-MM-DD
  - YYYY       -> period_type="duration", period_start=YYYY-01-01, period_end=YYYY-12-31
  - Other non-empty strings -> period_end defaults to the raw date string, type/start remain null.
- Remove suggested_period_type/suggested_period_start/suggested_period_end when present.

The migration is deterministic and idempotent.
"""

import re
from dataclasses import dataclass
from typing import Any, Iterable

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")

_PERIOD_KEYS = ("period_type", "period_start", "period_end")
_SUGGESTED_KEYS = ("suggested_period_type", "suggested_period_start", "suggested_period_end")


@dataclass
class MigrationStats:
    facts_scanned: int = 0
    facts_with_date: int = 0
    facts_updated: int = 0
    fields_added: int = 0
    fields_removed: int = 0
    suggestions_added: int = 0

    @property
    def changed(self) -> bool:
        return self.facts_updated > 0


def _normalize_date(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def suggest_period_fields(date_value: Any) -> tuple[str | None, str | None, str | None]:
    raw = _normalize_date(date_value)
    if not raw:
        return None, None, None
    if _DATE_RE.fullmatch(raw):
        return "instant", None, raw
    if _YEAR_RE.fullmatch(raw):
        return "duration", f"{raw}-01-01", f"{raw}-12-31"
    return None, None, None


def _looks_like_fact(item: dict[str, Any]) -> bool:
    return any(key in item for key in ("value", "date", "bbox", "currency", "scale", "value_type"))


def _iter_fact_dicts(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        facts = payload.get("facts")
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict):
                    yield fact

        pages = payload.get("pages")
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                page_facts = page.get("facts")
                if isinstance(page_facts, list):
                    for fact in page_facts:
                        if isinstance(fact, dict):
                            yield fact
                predictions = page.get("predictions")
                if isinstance(predictions, dict):
                    pred_facts = predictions.get("facts")
                    if isinstance(pred_facts, list):
                        for fact in pred_facts:
                            if isinstance(fact, dict):
                                yield fact

        predictions = payload.get("predictions")
        if isinstance(predictions, dict):
            pred_facts = predictions.get("facts")
            if isinstance(pred_facts, list):
                for fact in pred_facts:
                    if isinstance(fact, dict):
                        yield fact
        return

    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) and _looks_like_fact(item) for item in payload):
            for fact in payload:
                yield fact
            return
        for item in payload:
            yield from _iter_fact_dicts(item)


def migrate_fact(fact: dict[str, Any]) -> MigrationStats:
    stats = MigrationStats(facts_scanned=1)
    if "date" not in fact:
        return stats

    stats.facts_with_date = 1
    updated = False

    for key in _PERIOD_KEYS:
        if key not in fact:
            fact[key] = None
            stats.fields_added += 1
            updated = True

    for key in _SUGGESTED_KEYS:
        if key in fact:
            fact.pop(key, None)
            stats.fields_removed += 1
            updated = True

    suggested_type, suggested_start, suggested_end = suggest_period_fields(fact.get("date"))

    period_type_missing = fact.get("period_type") in (None, "")
    period_start_missing = fact.get("period_start") in (None, "")
    period_end_missing = fact.get("period_end") in (None, "")

    if suggested_type is not None and period_type_missing:
        fact["period_type"] = suggested_type
        stats.suggestions_added += 1
        updated = True
    if suggested_start is not None and period_start_missing:
        fact["period_start"] = suggested_start
        stats.suggestions_added += 1
        updated = True
    if suggested_end is not None and period_end_missing:
        fact["period_end"] = suggested_end
        stats.suggestions_added += 1
        updated = True

    if updated:
        stats.facts_updated = 1
    return stats


def migrate_payload(payload: Any) -> MigrationStats:
    stats = MigrationStats()
    for fact in _iter_fact_dicts(payload):
        fact_stats = migrate_fact(fact)
        stats.facts_scanned += fact_stats.facts_scanned
        stats.facts_with_date += fact_stats.facts_with_date
        stats.facts_updated += fact_stats.facts_updated
        stats.fields_added += fact_stats.fields_added
        stats.fields_removed += fact_stats.fields_removed
        stats.suggestions_added += fact_stats.suggestions_added
    return stats


__all__ = [
    "MigrationStats",
    "migrate_fact",
    "migrate_payload",
    "suggest_period_fields",
]
