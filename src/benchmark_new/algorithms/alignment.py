from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def align_sequences_by_index(
    ground_truth: list[Any],
    prediction: list[Any],
    *,
    missing_sentinel: Any = None,
) -> list[tuple[Any, Any]]:
    max_len = max(len(ground_truth), len(prediction))
    pairs: list[tuple[Any, Any]] = []
    for index in range(max_len):
        left = ground_truth[index] if index < len(ground_truth) else missing_sentinel
        right = prediction[index] if index < len(prediction) else missing_sentinel
        pairs.append((left, right))
    return pairs


def build_row_diff_diagnostics(ground_truth_rows: list[str], prediction_rows: list[str]) -> list[dict[str, Any]]:
    matcher = SequenceMatcher(None, ground_truth_rows, prediction_rows)
    diagnostics: list[dict[str, Any]] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        diagnostics.append(
            {
                "op": opcode,
                "gt_start": i1,
                "gt_end": i2,
                "pred_start": j1,
                "pred_end": j2,
                "gt_rows": ground_truth_rows[i1:i2],
                "pred_rows": prediction_rows[j1:j2],
            }
        )
    return diagnostics
