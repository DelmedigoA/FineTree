from __future__ import annotations

import math
from typing import Literal

_MIN_EVEN_SPAN = 2


def _floor_even(value: float) -> int:
    return int(math.floor(float(value) / 2.0)) * 2


def _ceil_even(value: float) -> int:
    return int(math.ceil(float(value) / 2.0)) * 2


def _nearest_even(value: float) -> int:
    lower = _floor_even(value)
    upper = _ceil_even(value)
    lower_distance = abs(float(value) - float(lower))
    upper_distance = abs(float(upper) - float(value))
    if lower_distance < upper_distance:
        return lower
    if upper_distance < lower_distance:
        return upper
    return upper


def _outward_even(value: float, *, edge: Literal["min", "max"]) -> int:
    nearest = _nearest_even(value)
    if edge == "min":
        return nearest if float(nearest) <= float(value) else (nearest - 2)
    return nearest if float(nearest) >= float(value) else (nearest + 2)


def snap_even_edge_pair(
    min_value: float,
    max_value: float,
    *,
    lower_bound: float,
    upper_bound: float,
    min_span: int = _MIN_EVEN_SPAN,
) -> tuple[int, int, bool]:
    min_span = max(int(min_span), _MIN_EVEN_SPAN)
    lower_even = _ceil_even(lower_bound)
    upper_even = _floor_even(upper_bound)
    if (upper_even - lower_even) < min_span:
        raise ValueError(
            "Cannot quantize bbox edges to a positive even span within bounds "
            f"{lower_bound}..{upper_bound}."
        )

    ideal_min = _outward_even(min_value, edge="min")
    ideal_max = _outward_even(max_value, edge="max")
    snapped_min = max(lower_even, ideal_min)
    snapped_max = min(upper_even, ideal_max)
    adjusted = (snapped_min != ideal_min) or (snapped_max != ideal_max)

    if (snapped_max - snapped_min) >= min_span:
        return snapped_min, snapped_max, adjusted

    adjusted = True
    if snapped_max >= upper_even:
        snapped_max = upper_even
        snapped_min = snapped_max - min_span
    elif snapped_min <= lower_even:
        snapped_min = lower_even
        snapped_max = snapped_min + min_span
    else:
        snapped_max = min(upper_even, snapped_min + min_span)
        snapped_min = snapped_max - min_span
        if snapped_min < lower_even:
            snapped_min = lower_even
            snapped_max = snapped_min + min_span

    return snapped_min, snapped_max, adjusted


def quantize_xywh_bbox_to_even(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[list[int], bool]:
    left = float(x)
    top = float(y)
    right = float(x) + float(w)
    bottom = float(y) + float(h)
    left_i, right_i, x_adjusted = snap_even_edge_pair(
        left,
        right,
        lower_bound=0.0,
        upper_bound=float(image_width),
    )
    top_i, bottom_i, y_adjusted = snap_even_edge_pair(
        top,
        bottom,
        lower_bound=0.0,
        upper_bound=float(image_height),
    )
    return [left_i, top_i, right_i - left_i, bottom_i - top_i], (x_adjusted or y_adjusted)


def quantize_yxyx_bbox_to_even(
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    *,
    upper_bound: int = 1000,
) -> tuple[list[int], bool]:
    ymin_i, ymax_i, y_adjusted = snap_even_edge_pair(
        ymin,
        ymax,
        lower_bound=0.0,
        upper_bound=float(upper_bound),
    )
    xmin_i, xmax_i, x_adjusted = snap_even_edge_pair(
        xmin,
        xmax,
        lower_bound=0.0,
        upper_bound=float(upper_bound),
    )
    return [ymin_i, xmin_i, ymax_i, xmax_i], (x_adjusted or y_adjusted)


__all__ = [
    "quantize_xywh_bbox_to_even",
    "quantize_yxyx_bbox_to_even",
    "snap_even_edge_pair",
]
