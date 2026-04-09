from __future__ import annotations

from statistics import mean


def numeric_closeness_details(pairs: list[tuple[float | None, float | None]]) -> dict[str, float | int | None]:
    scores: list[float] = []
    gt_values: list[float] = []
    pred_values: list[float] = []
    for gt_value, pred_value in pairs:
        if gt_value is None or pred_value is None:
            continue
        gt_values.append(gt_value)
        pred_values.append(pred_value)
        scores.append(max(0.0, 1.0 - abs(pred_value - gt_value) / max(abs(gt_value), 1.0)))
    mae = None
    if gt_values and pred_values:
        try:
            from sklearn.metrics import mean_absolute_error

            mae = float(mean_absolute_error(gt_values, pred_values))
        except Exception:
            mae = float(mean(abs(pred - gt) for gt, pred in zip(gt_values, pred_values, strict=False)))
    return {
        "score": float(mean(scores)) if scores else None,
        "pair_count": len(scores),
        "mae": mae,
    }
