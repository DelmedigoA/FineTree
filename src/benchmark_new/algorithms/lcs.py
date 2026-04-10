from __future__ import annotations

from typing import Any

from ..models import FieldExactMetrics


def _lcs_table(left: list[Any], right: list[Any]) -> list[list[int]]:
    rows = len(left)
    cols = len(right)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for left_index in range(rows):
        for right_index in range(cols):
            if left[left_index] == right[right_index]:
                table[left_index + 1][right_index + 1] = table[left_index][right_index] + 1
            else:
                table[left_index + 1][right_index + 1] = max(
                    table[left_index][right_index + 1],
                    table[left_index + 1][right_index],
                )
    return table


def lcs_length(left: list[Any], right: list[Any]) -> int:
    if not left or not right:
        return 0
    table = _lcs_table(left, right)
    return table[len(left)][len(right)]


def lcs_index_pairs(left: list[Any], right: list[Any]) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    table = _lcs_table(left, right)
    left_index = len(left)
    right_index = len(right)
    pairs: list[tuple[int, int]] = []
    while left_index > 0 and right_index > 0:
        if left[left_index - 1] == right[right_index - 1]:
            pairs.append((left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
            continue
        if table[left_index - 1][right_index] >= table[left_index][right_index - 1]:
            left_index -= 1
        else:
            right_index -= 1
    pairs.reverse()
    return pairs


def lcs_metrics(ground_truth: list[Any], prediction: list[Any]) -> FieldExactMetrics:
    matches = lcs_length(ground_truth, prediction)
    if not ground_truth and not prediction:
        precision = 1.0
        recall = 1.0
    elif not prediction:
        precision = 1.0
        recall = 0.0
    elif not ground_truth:
        precision = 0.0
        recall = 1.0
    else:
        precision = float(matches / len(prediction))
        recall = float(matches / len(ground_truth))
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    denominator = max(len(ground_truth), len(prediction))
    accuracy = float(matches / denominator) if denominator else 1.0
    return FieldExactMetrics(
        matches=matches,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )
