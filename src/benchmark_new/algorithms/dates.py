from __future__ import annotations

from statistics import mean
from datetime import date


def date_diff_details(pairs: list[tuple[date | None, date | None]]) -> dict[str, float | int | None]:
    scores: list[float] = []
    diffs: list[int] = []
    for gt_value, pred_value in pairs:
        if gt_value is None or pred_value is None:
            continue
        day_diff = abs((pred_value - gt_value).days)
        diffs.append(day_diff)
        scores.append(max(0.0, 1.0 - (day_diff / 30.0)))
    return {
        "score": float(mean(scores)) if scores else None,
        "pair_count": len(scores),
        "mae_days": float(mean(diffs)) if diffs else None,
    }
