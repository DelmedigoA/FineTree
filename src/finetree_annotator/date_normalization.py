from __future__ import annotations

import re
from datetime import date
from typing import Any

_DATE_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_YM_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")
_DATE_YMD_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$")
_DATE_DMY_RE = re.compile(r"^(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})$")


def _to_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_date(raw_date: Any) -> tuple[str | None, list[str]]:
    text = _to_optional_text(raw_date)
    if text is None:
        return None, []

    if _DATE_YEAR_RE.match(text):
        return text, []

    m_ym = _DATE_YM_RE.match(text)
    if m_ym:
        month = int(m_ym.group("month"))
        if 1 <= month <= 12:
            return text, []
        return text, ["noncanonical_date"]

    m_ymd = _DATE_YMD_RE.match(text)
    if m_ymd:
        try:
            date(
                int(m_ymd.group("year")),
                int(m_ymd.group("month")),
                int(m_ymd.group("day")),
            )
        except ValueError:
            return text, ["noncanonical_date"]
        return text, []

    m_dmy = _DATE_DMY_RE.match(text)
    if m_dmy:
        try:
            parsed = date(
                int(m_dmy.group("year")),
                int(m_dmy.group("month")),
                int(m_dmy.group("day")),
            )
        except ValueError:
            return text, ["noncanonical_date"]
        return parsed.strftime("%Y-%m-%d"), []

    return text, ["noncanonical_date"]


__all__ = ["normalize_date"]
