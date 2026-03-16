from __future__ import annotations

from typing import Any, Optional


def normalize_bbox_data(data: Optional[Any]) -> dict[str, float]:
    if isinstance(data, dict):
        x_raw = data.get("x", 0.0)
        y_raw = data.get("y", 0.0)
        w_raw = data.get("w", 1.0)
        h_raw = data.get("h", 1.0)
    elif isinstance(data, (list, tuple)) and len(data) >= 4:
        x_raw, y_raw, w_raw, h_raw = data[0], data[1], data[2], data[3]
    else:
        x_raw, y_raw, w_raw, h_raw = 0.0, 0.0, 1.0, 1.0

    x = float(x_raw)
    y = float(y_raw)
    w = max(float(w_raw), 1.0)
    h = max(float(h_raw), 1.0)
    return {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}


def bbox_to_list(data: Optional[Any]) -> list[float]:
    bbox = normalize_bbox_data(data)
    return [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]


def denormalize_bbox_from_1000(data: Optional[dict[str, Any]], image_width: float, image_height: float) -> dict[str, float]:
    bbox = normalize_bbox_data(data)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    return normalize_bbox_data(
        {
            "x": (bbox["x"] * width) / 1000.0,
            "y": (bbox["y"] * height) / 1000.0,
            "w": (bbox["w"] * width) / 1000.0,
            "h": (bbox["h"] * height) / 1000.0,
        }
    )


def bbox_looks_normalized_1000(bbox: dict[str, Any]) -> bool:
    try:
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("w", 0.0))
        h = float(bbox.get("h", 0.0))
    except Exception:
        return False
    limit = 1000.0 + 1e-6
    return (
        0.0 <= x <= limit
        and 0.0 <= y <= limit
        and 0.0 <= w <= limit
        and 0.0 <= h <= limit
        and (x + w) <= limit
        and (y + h) <= limit
    )


__all__ = [
    "bbox_looks_normalized_1000",
    "bbox_to_list",
    "denormalize_bbox_from_1000",
    "normalize_bbox_data",
]
