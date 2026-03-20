from __future__ import annotations

import inspect
import math
from typing import Any, Optional

from .bbox_utils import normalize_bbox_data

DEFAULT_QWEN_VISION_FACTOR = 28
_MIN_BBOX_SIZE = 1.0


def fallback_smart_resize_dimensions(
    height: int,
    width: int,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
    factor: int = DEFAULT_QWEN_VISION_FACTOR,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions.")

    pixels = int(height) * int(width)
    scale = 1.0
    snap_mode = "nearest"
    if min_pixels is not None and pixels < int(min_pixels):
        scale = max(scale, (int(min_pixels) / float(pixels)) ** 0.5)
        snap_mode = "ceil"
    if max_pixels is not None and pixels > int(max_pixels):
        scale = (
            min(scale, (int(max_pixels) / float(pixels)) ** 0.5)
            if scale != 1.0
            else (int(max_pixels) / float(pixels)) ** 0.5
        )
        snap_mode = "floor"

    target_h = max(1, int(round(int(height) * scale)))
    target_w = max(1, int(round(int(width) * scale)))
    if factor > 1:
        if snap_mode == "ceil":
            target_h = max(factor, int(math.ceil(target_h / float(factor))) * factor)
            target_w = max(factor, int(math.ceil(target_w / float(factor))) * factor)
        else:
            target_h = max(factor, (target_h // factor) * factor)
            target_w = max(factor, (target_w // factor) * factor)

    if min_pixels is not None:
        while target_h * target_w < int(min_pixels):
            grew = False
            if target_w <= target_h:
                target_w += factor
                grew = True
            if target_h * target_w < int(min_pixels):
                target_h += factor
                grew = True
            if not grew:
                break

    if max_pixels is not None:
        while target_h * target_w > int(max_pixels) and (target_h > factor or target_w > factor):
            if target_w >= target_h and target_w > factor:
                target_w -= factor
            elif target_h > factor:
                target_h -= factor
            else:
                break

    return int(target_h), int(target_w)


def smart_resize_dimensions(
    height: int,
    width: int,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    try:
        from qwen_vl_utils.vision_process import smart_resize
    except Exception:
        return fallback_smart_resize_dimensions(height, width, min_pixels=min_pixels, max_pixels=max_pixels)

    kwargs: dict[str, int] = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
    sig = inspect.signature(smart_resize)
    factor_param = sig.parameters.get("factor")
    if factor_param is not None and factor_param.default is inspect._empty:
        kwargs["factor"] = DEFAULT_QWEN_VISION_FACTOR
    try:
        new_h, new_w = smart_resize(int(height), int(width), **kwargs)
    except Exception:
        return fallback_smart_resize_dimensions(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    return int(new_h), int(new_w)


def _clamp_bbox_to_image_bounds(
    bbox: Optional[Any],
    *,
    image_width: float,
    image_height: float,
) -> tuple[dict[str, float], bool]:
    normalized = normalize_bbox_data(bbox)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)

    x = max(0.0, float(normalized["x"]))
    y = max(0.0, float(normalized["y"]))
    max_x = max(width - _MIN_BBOX_SIZE, 0.0)
    max_y = max(height - _MIN_BBOX_SIZE, 0.0)
    x = min(x, max_x)
    y = min(y, max_y)

    w = max(float(normalized["w"]), _MIN_BBOX_SIZE)
    h = max(float(normalized["h"]), _MIN_BBOX_SIZE)
    max_w = max(width - x, _MIN_BBOX_SIZE)
    max_h = max(height - y, _MIN_BBOX_SIZE)
    w = min(w, max_w)
    h = min(h, max_h)

    clamped = (
        x != float(normalized["x"])
        or y != float(normalized["y"])
        or w != float(normalized["w"])
        or h != float(normalized["h"])
    )
    return normalize_bbox_data({"x": x, "y": y, "w": w, "h": h}), clamped


def _prepared_dimensions_for_resize_restore(
    *,
    original_width: float,
    original_height: float,
    max_pixels: int,
) -> tuple[int, int]:
    width = max(int(round(float(original_width))), 1)
    height = max(int(round(float(original_height))), 1)
    if width * height <= int(max_pixels):
        return height, width
    return smart_resize_dimensions(height, width, min_pixels=None, max_pixels=int(max_pixels))


def prepared_dimensions_for_max_pixels(
    *,
    original_width: float,
    original_height: float,
    max_pixels: int,
) -> tuple[int, int]:
    if int(max_pixels) <= 0:
        raise ValueError("max_pixels must be > 0.")
    return _prepared_dimensions_for_resize_restore(
        original_width=original_width,
        original_height=original_height,
        max_pixels=int(max_pixels),
    )


def restore_bbox_from_resized_pixels_with_stats(
    bbox: Optional[Any],
    *,
    original_width: float,
    original_height: float,
    max_pixels: int,
) -> tuple[dict[str, float], bool]:
    if int(max_pixels) <= 0:
        raise ValueError("max_pixels must be > 0.")

    prepared_h, prepared_w = _prepared_dimensions_for_resize_restore(
        original_width=original_width,
        original_height=original_height,
        max_pixels=int(max_pixels),
    )
    scale_x = max(float(original_width), 1.0) / max(float(prepared_w), 1.0)
    scale_y = max(float(original_height), 1.0) / max(float(prepared_h), 1.0)
    normalized = normalize_bbox_data(bbox)
    restored = normalize_bbox_data(
        {
            "x": float(normalized["x"]) * scale_x,
            "y": float(normalized["y"]) * scale_y,
            "w": float(normalized["w"]) * scale_x,
            "h": float(normalized["h"]) * scale_y,
        }
    )
    return _clamp_bbox_to_image_bounds(
        restored,
        image_width=original_width,
        image_height=original_height,
    )


def restore_bbox_from_resized_pixels(
    bbox: Optional[Any],
    *,
    original_width: float,
    original_height: float,
    max_pixels: int,
) -> dict[str, float]:
    restored, _clamped = restore_bbox_from_resized_pixels_with_stats(
        bbox,
        original_width=original_width,
        original_height=original_height,
        max_pixels=max_pixels,
    )
    return restored


__all__ = [
    "DEFAULT_QWEN_VISION_FACTOR",
    "fallback_smart_resize_dimensions",
    "prepared_dimensions_for_max_pixels",
    "restore_bbox_from_resized_pixels",
    "restore_bbox_from_resized_pixels_with_stats",
    "smart_resize_dimensions",
]
