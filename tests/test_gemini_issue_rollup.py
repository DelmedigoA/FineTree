from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_issue_summary(session_dir: Path, payload: dict[str, object]) -> None:
    session_dir.mkdir(parents=True)
    (session_dir / "issue_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_gemini_issue_rollup_groups_recurring_signatures(tmp_path: Path) -> None:
    logs_dir = tmp_path / "gemini_logs"
    shared = {
        "model": "gemini-3-flash-preview",
        "operation": "stream_content",
        "statement_type": "notes_to_financial_statements",
        "prompt_signature": "prompt-a",
        "few_shot_signature": "few-a",
        "issue_signature": "sig-a",
        "issue_count": 24,
        "issue_category_counts": {"schema": 24},
        "validation_failure_groups": [
            {"code": "note_num_without_note_flag", "count": 24},
        ],
    }
    _write_issue_summary(
        logs_dir / "20260319T115800.605745Z_stream_content_gemini-3-flash-preview",
        {"session_id": "s1", **shared},
    )
    _write_issue_summary(
        logs_dir / "20260319T120000.000000Z_stream_content_gemini-3-flash-preview",
        {"session_id": "s2", **shared},
    )
    _write_issue_summary(
        logs_dir / "20260319T121000.000000Z_stream_content_gemini-3-flash-preview",
        {
            "session_id": "s3",
            "model": "gemini-3-flash-preview",
            "operation": "stream_content",
            "statement_type": "income_statement",
            "prompt_signature": "prompt-b",
            "few_shot_signature": "few-b",
            "issue_signature": "sig-b",
            "issue_count": 1,
            "issue_category_counts": {"bbox": 1},
            "validation_failure_groups": [
                {"code": "bbox_repeated_strip_pattern", "count": 1},
            ],
        },
    )

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/gemini_issue_rollup.py"),
            "--logs-dir",
            str(logs_dir),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["session_count"] == 3
    assert payload["group_count"] == 2
    first_group = payload["groups"][0]
    assert first_group["issue_signature"] == "sig-a"
    assert first_group["session_count"] == 2
    assert first_group["issue_codes"]["note_num_without_note_flag"] == 48
