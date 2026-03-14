from __future__ import annotations

import pytest

from finetree_annotator.benchmark.logging_summary import parse_logging_jsonl_bytes, summarize_logging_rows


def test_logging_summary_extracts_latest_and_best_metrics() -> None:
    raw = "\n".join(
        [
            '{"loss": 0.5, "learning_rate": 0.001, "token_acc": 0.9, "epoch": 1.0, "global_step/max_steps": "1/4", "memory(GiB)": 10.5, "train_speed(s/it)": 2.5}',
            '{"eval_loss": 0.4, "eval_runtime": 11.0, "eval_token_acc": 0.91, "epoch": 1.0, "global_step/max_steps": "1/4", "elapsed_time": "1m", "remaining_time": "3m", "memory(GiB)": 11.0}',
            '{"loss": 0.3, "learning_rate": 0.0005, "token_acc": 0.94, "epoch": 2.0, "global_step/max_steps": "2/4", "memory(GiB)": 12.25, "train_speed(s/it)": 2.0}',
            '{"eval_loss": 0.2, "eval_runtime": 10.5, "eval_token_acc": 0.96, "epoch": 2.0, "global_step/max_steps": "2/4", "elapsed_time": "2m", "remaining_time": "2m", "memory(GiB)": 12.0}',
        ]
    )
    _text, rows = parse_logging_jsonl_bytes(raw.encode("utf-8"))
    summary = summarize_logging_rows(rows)
    assert summary["row_count"] == 4
    assert summary["latest_train_loss"] == 0.3
    assert summary["latest_eval_loss"] == 0.2
    assert summary["best_eval_loss"] == 0.2
    assert summary["latest_token_accuracy"] == 0.94
    assert summary["latest_eval_token_accuracy"] == 0.96
    assert summary["latest_global_step"] == 2
    assert summary["latest_max_steps"] == 4
    assert summary["max_memory_gib"] == 12.25
    assert summary["latest_train_speed"] == 2.0
    assert summary["latest_learning_rate"] == 0.0005
    assert summary["latest_eval_runtime"] == 10.5


def test_parse_logging_jsonl_rejects_invalid_lines() -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_logging_jsonl_bytes(b'{"loss": 1}\nnot-json\n')
