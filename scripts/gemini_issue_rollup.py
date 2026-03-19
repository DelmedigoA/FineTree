from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_issue_summaries(logs_dir: Path, *, limit: int | None) -> list[dict[str, Any]]:
    session_dirs = sorted(
        [path for path in logs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    if limit is not None and limit > 0:
        session_dirs = session_dirs[:limit]

    summaries: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        summary_path = session_dir / "issue_summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["session_dir"] = str(session_dir)
        summaries.append(payload)
    return summaries


def build_rollup(logs_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    summaries = _load_issue_summaries(logs_dir, limit=limit)
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for summary in summaries:
        key = (
            str(summary.get("model") or ""),
            str(summary.get("operation") or ""),
            str(summary.get("statement_type") or ""),
            str(summary.get("prompt_signature") or ""),
            str(summary.get("few_shot_signature") or ""),
            str(summary.get("issue_signature") or ""),
        )
        group = grouped.setdefault(
            key,
            {
                "model": key[0],
                "operation": key[1],
                "statement_type": key[2],
                "prompt_signature": key[3],
                "few_shot_signature": key[4],
                "issue_signature": key[5],
                "session_count": 0,
                "issue_count": 0,
                "issue_category_counts": defaultdict(int),
                "issue_codes": defaultdict(int),
                "session_ids": [],
            },
        )
        group["session_count"] += 1
        group["issue_count"] += int(summary.get("issue_count") or 0)
        session_id = str(summary.get("session_id") or "")
        if session_id:
            group["session_ids"].append(session_id)
        for category, count in (summary.get("issue_category_counts") or {}).items():
            group["issue_category_counts"][str(category)] += int(count or 0)
        for failure_group in summary.get("validation_failure_groups") or []:
            if not isinstance(failure_group, dict):
                continue
            code = str(failure_group.get("code") or "").strip()
            if not code:
                continue
            group["issue_codes"][code] += int(failure_group.get("count") or 0)

    group_list: list[dict[str, Any]] = []
    for group in grouped.values():
        issue_category_counts = dict(sorted(group["issue_category_counts"].items()))
        issue_codes = dict(sorted(group["issue_codes"].items(), key=lambda item: (-item[1], item[0])))
        group_list.append(
            {
                **{key: value for key, value in group.items() if key not in {"issue_category_counts", "issue_codes"}},
                "issue_category_counts": issue_category_counts,
                "issue_codes": issue_codes,
            }
        )

    group_list.sort(key=lambda item: (-int(item["session_count"]), -int(item["issue_count"]), item["issue_signature"]))
    return {
        "logs_dir": str(logs_dir),
        "session_count": len(summaries),
        "group_count": len(group_list),
        "groups": group_list,
    }


def _format_rollup_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Logs: {payload.get('logs_dir')}",
        f"Sessions: {int(payload.get('session_count') or 0)}",
        f"Groups: {int(payload.get('group_count') or 0)}",
    ]
    for group in payload.get("groups") or []:
        lines.append(
            (
                f"- {group.get('model')} | {group.get('operation')} | "
                f"statement_type={group.get('statement_type') or 'unknown'} | "
                f"sessions={int(group.get('session_count') or 0)} | "
                f"issues={int(group.get('issue_count') or 0)} | "
                f"issue_signature={group.get('issue_signature')}"
            )
        )
        if group.get("issue_codes"):
            issue_codes = ", ".join(
                f"{code}:{count}"
                for code, count in list((group.get("issue_codes") or {}).items())[:5]
            )
            lines.append(f"  issue_codes={issue_codes}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Gemini issue_summary.json artifacts across gemini_logs sessions.")
    parser.add_argument("--logs-dir", default="gemini_logs", help="Directory containing Gemini session log folders.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of newest session folders to scan.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    payload = build_rollup(Path(args.logs_dir), limit=args.limit)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_rollup_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
