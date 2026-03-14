from __future__ import annotations

import json
import math
from typing import Any


def parse_logging_jsonl_bytes(raw_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("logging.jsonl must be valid UTF-8.") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"logging.jsonl line {line_number} is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"logging.jsonl line {line_number} must contain a JSON object.")
        rows.append(payload)
    if not rows:
        raise ValueError("logging.jsonl must contain at least one JSON object row.")
    return text, rows


def _latest_row_with_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if key in row and row.get(key) not in (None, ""):
            return row
    return None


def _all_numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or value in (None, ""):
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            values.append(parsed)
    return values


def _latest_numeric_value(rows: list[dict[str, Any]], key: str) -> float | None:
    row = _latest_row_with_key(rows, key)
    if row is None:
        return None
    try:
        return float(row[key])
    except Exception:
        return None


def _parse_step_progress(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if "/" not in text:
        return {"raw": text or None, "global_step": None, "max_steps": None, "progress_ratio": None}
    left, right = text.split("/", 1)
    try:
        global_step = int(left)
        max_steps = int(right)
    except ValueError:
        return {"raw": text or None, "global_step": None, "max_steps": None, "progress_ratio": None}
    ratio = None
    if max_steps > 0:
        ratio = global_step / float(max_steps)
    return {
        "raw": text,
        "global_step": global_step,
        "max_steps": max_steps,
        "progress_ratio": ratio,
    }


def summarize_logging_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_train_row = _latest_row_with_key(rows, "loss")
    latest_eval_row = _latest_row_with_key(rows, "eval_loss")
    latest_progress_row = _latest_row_with_key(rows, "global_step/max_steps")
    eval_losses = _all_numeric_values(rows, "eval_loss")
    memory_values = _all_numeric_values(rows, "memory(GiB)")
    summary = {
        "row_count": len(rows),
        "train_row_count": sum(1 for row in rows if "loss" in row),
        "eval_row_count": sum(1 for row in rows if any(str(key).startswith("eval_") for key in row.keys())),
        "available_keys": sorted({str(key) for row in rows for key in row.keys()}),
        "latest_train_loss": _latest_numeric_value(rows, "loss"),
        "latest_eval_loss": _latest_numeric_value(rows, "eval_loss"),
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "latest_token_accuracy": _latest_numeric_value(rows, "token_acc"),
        "latest_eval_token_accuracy": _latest_numeric_value(rows, "eval_token_acc"),
        "max_observed_epoch": max(_all_numeric_values(rows, "epoch"), default=None),
        "latest_learning_rate": _latest_numeric_value(rows, "learning_rate"),
        "latest_eval_runtime": _latest_numeric_value(rows, "eval_runtime"),
        "max_memory_gib": max(memory_values, default=None),
        "latest_train_speed": _latest_numeric_value(rows, "train_speed(s/it)"),
        "latest_elapsed_time": latest_progress_row.get("elapsed_time") if latest_progress_row else None,
        "latest_remaining_time": latest_progress_row.get("remaining_time") if latest_progress_row else None,
        "latest_train_row": latest_train_row,
        "latest_eval_row": latest_eval_row,
    }
    step_progress = _parse_step_progress(
        latest_progress_row.get("global_step/max_steps") if latest_progress_row else None
    )
    summary.update(
        {
            "latest_step_progress": step_progress.get("raw"),
            "latest_global_step": step_progress.get("global_step"),
            "latest_max_steps": step_progress.get("max_steps"),
            "latest_step_progress_ratio": step_progress.get("progress_ratio"),
        }
    )
    return summary
