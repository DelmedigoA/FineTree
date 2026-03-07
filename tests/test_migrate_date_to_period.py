from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.migrations.date_to_period import migrate_payload, suggest_period_fields


def test_suggest_period_fields_for_ymd() -> None:
    assert suggest_period_fields("2024-12-31") == ("instant", None, "2024-12-31")


def test_suggest_period_fields_for_year() -> None:
    assert suggest_period_fields("2024") == ("duration", "2024-01-01", "2024-12-31")


def test_migrate_payload_adds_period_and_suggestions_for_date() -> None:
    payload = {
        "pages": [
            {
                "facts": [
                    {
                        "value": "1,645",
                        "date": "2024-12-31",
                    }
                ]
            }
        ]
    }
    stats = migrate_payload(payload)
    fact = payload["pages"][0]["facts"][0]
    assert stats.facts_with_date == 1
    assert fact["date"] == "2024-12-31"
    assert fact["period_type"] == "instant"
    assert fact["period_start"] is None
    assert fact["period_end"] == "2024-12-31"
    assert "suggested_period_type" not in fact
    assert "suggested_period_start" not in fact
    assert "suggested_period_end" not in fact


def test_migrate_payload_handles_year_only_date() -> None:
    payload = {
        "pages": [
            {
                "facts": [
                    {
                        "value": "1,645",
                        "date": "2024",
                    }
                ]
            }
        ]
    }
    migrate_payload(payload)
    fact = payload["pages"][0]["facts"][0]
    assert fact["period_type"] == "duration"
    assert fact["period_start"] == "2024-01-01"
    assert fact["period_end"] == "2024-12-31"
    assert "suggested_period_type" not in fact


def test_migrate_payload_preserves_existing_fields() -> None:
    payload = {
        "pages": [
            {
                "facts": [
                    {
                        "value": "1,645",
                        "date": "2024-12-31",
                        "period_type": "duration",
                        "period_start": "2024-01-01",
                        "period_end": "2024-12-31",
                    }
                ]
            }
        ]
    }
    migrate_payload(payload)
    fact = payload["pages"][0]["facts"][0]
    assert fact["period_type"] == "duration"
    assert fact["period_start"] == "2024-01-01"
    assert fact["period_end"] == "2024-12-31"
    assert "suggested_period_type" not in fact
    assert "suggested_period_start" not in fact
    assert "suggested_period_end" not in fact


def test_migrate_payload_is_idempotent() -> None:
    payload = {
        "pages": [
            {
                "facts": [
                    {
                        "value": "1,645",
                        "date": "2024",
                    }
                ]
            }
        ]
    }
    migrate_payload(payload)
    snapshot = json.dumps(payload, sort_keys=True)
    stats = migrate_payload(payload)
    assert json.dumps(payload, sort_keys=True) == snapshot
    assert stats.facts_updated == 0


def test_migrate_payload_handles_top_level_facts() -> None:
    payload = {"facts": [{"value": "10", "date": "2024-12-31"}]}
    migrate_payload(payload)
    fact = payload["facts"][0]
    assert fact["period_type"] == "instant"
    assert fact["period_end"] == "2024-12-31"


def test_migrate_payload_handles_other_date_formats() -> None:
    payload = {"facts": [{"value": "10", "date": "2024-12"}]}
    migrate_payload(payload)
    fact = payload["facts"][0]
    assert fact["period_type"] is None
    assert fact["period_start"] is None
    assert fact["period_end"] is None
