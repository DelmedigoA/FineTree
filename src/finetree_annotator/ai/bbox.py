from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtGui import QImage

from ..annotation_core import normalize_fact_data
from ..bbox_utils import bbox_to_list, denormalize_bbox_from_1000, normalize_bbox_data
from ..fact_ordering import canonical_fact_order_indices

BBOX_MODE_PIXEL_AS_IS = "pixel_as_is"
BBOX_MODE_NORMALIZED_1000_TO_PIXEL = "normalized_1000_to_pixel"
BBOX_MODE_MIXED_AUTO = "mixed_auto"
BBOX_MODE_SWITCH_MARGIN = 0.08
BBOX_PAGE_MODE_LOCK_MARGIN = 0.2
BBOX_DARK_LUMA_THRESHOLD = 220.0
BBOX_INK_TARGET_RATIO = 0.12


@dataclass(frozen=True)
class BBoxResolutionResult:
    payloads: list[dict[str, Any]]
    mode: str
    scores: dict[str, float]
    fact_modes: list[str]
    policy: str


def normalize_bbox_to_list(raw_bbox: Any) -> list[float]:
    return bbox_to_list(raw_bbox)


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


def normalize_ai_fact_payload(
    fact_payload: dict[str, Any],
    *,
    clear_fact_num: bool = False,
) -> dict[str, Any] | None:
    raw_bbox = fact_payload.get("bbox")
    if not isinstance(raw_bbox, dict) and not (isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4):
        return None
    bbox = normalize_bbox_data(raw_bbox)
    fact_data = normalize_fact_data(fact_payload)
    if clear_fact_num:
        fact_data = normalize_fact_data({**fact_data, "fact_num": None})
    return {"bbox": bbox_to_list(bbox), **fact_data}


def payloads_for_bbox_mode(
    fact_payloads: list[dict[str, Any]],
    *,
    mode: str,
    image_width: float,
    image_height: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in fact_payloads:
        normalized_payload = normalize_ai_fact_payload(payload)
        if normalized_payload is None:
            continue
        bbox = normalize_bbox_data(normalized_payload.get("bbox"))
        if mode == BBOX_MODE_NORMALIZED_1000_TO_PIXEL:
            bbox = denormalize_bbox_from_1000(bbox, image_width, image_height)
        out.append({"bbox": bbox_to_list(bbox), **normalize_fact_data(normalized_payload)})
    return out


def score_bbox_payload_ink(image: QImage, bbox_payload: dict[str, Any]) -> tuple[float, float]:
    image_width = image.width()
    image_height = image.height()
    if image_width <= 0 or image_height <= 0:
        return 0.0, 0.0
    bbox = normalize_bbox_data(bbox_payload.get("bbox"))
    area = max(float(bbox["w"]) * float(bbox["h"]), 1.0)
    left = float(bbox["x"])
    top = float(bbox["y"])
    right = left + float(bbox["w"])
    bottom = top + float(bbox["h"])

    sample_left = max(0, min(image_width - 1, int(math.floor(left))))
    sample_top = max(0, min(image_height - 1, int(math.floor(top))))
    sample_right = max(0, min(image_width, int(math.ceil(right))))
    sample_bottom = max(0, min(image_height, int(math.ceil(bottom))))
    if sample_right <= sample_left or sample_bottom <= sample_top:
        return 0.0, 0.0

    clipped_area = float((sample_right - sample_left) * (sample_bottom - sample_top))
    coverage = max(0.0, min(1.0, clipped_area / area))
    span_w = sample_right - sample_left
    span_h = sample_bottom - sample_top
    step = max(1, int(max(span_w / 32.0, span_h / 32.0)))

    dark_pixels = 0
    sampled_pixels = 0
    for py in range(sample_top, sample_bottom, step):
        for px in range(sample_left, sample_right, step):
            color = image.pixelColor(px, py)
            luma = 0.299 * float(color.red()) + 0.587 * float(color.green()) + 0.114 * float(color.blue())
            if luma <= BBOX_DARK_LUMA_THRESHOLD:
                dark_pixels += 1
            sampled_pixels += 1
    if sampled_pixels <= 0:
        return 0.0, coverage

    ink_ratio = dark_pixels / sampled_pixels
    ink_score = max(0.0, min(1.0, ink_ratio / BBOX_INK_TARGET_RATIO))
    score = (0.75 * ink_score) + (0.25 * coverage)
    if coverage < 0.35:
        score *= coverage / 0.35
    return score, coverage


def score_bbox_candidate_payloads(image_path: Path, fact_payloads: list[dict[str, Any]]) -> float:
    image = QImage(str(image_path))
    if image.isNull() or not fact_payloads:
        return 0.0

    total_score = 0.0
    total_coverage = 0.0
    scored_count = 0
    for payload in fact_payloads:
        score, coverage = score_bbox_payload_ink(image, payload)
        total_score += score
        total_coverage += coverage
        scored_count += 1
    if scored_count <= 0:
        return 0.0
    avg_score = total_score / float(scored_count)
    avg_coverage = total_coverage / float(scored_count)
    return max(0.0, avg_score - ((1.0 - avg_coverage) * 0.35))


def resolve_bbox_mode(
    *,
    image_path: Path,
    image_dimensions: Optional[tuple[float, float]],
    fact_payloads: list[dict[str, Any]],
) -> BBoxResolutionResult:
    empty_scores = {
        BBOX_MODE_PIXEL_AS_IS: 0.0,
        BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
    }
    if not fact_payloads:
        return BBoxResolutionResult(
            payloads=[],
            mode=BBOX_MODE_PIXEL_AS_IS,
            scores=empty_scores,
            fact_modes=[],
            policy="page_locked",
        )
    if image_dimensions is None:
        return BBoxResolutionResult(
            payloads=[deepcopy(payload) for payload in fact_payloads],
            mode=BBOX_MODE_PIXEL_AS_IS,
            scores=empty_scores,
            fact_modes=[BBOX_MODE_PIXEL_AS_IS for _ in fact_payloads],
            policy="page_locked",
        )

    image_width, image_height = image_dimensions
    pixel_payloads = payloads_for_bbox_mode(
        fact_payloads,
        mode=BBOX_MODE_PIXEL_AS_IS,
        image_width=image_width,
        image_height=image_height,
    )
    normalized_payloads = payloads_for_bbox_mode(
        fact_payloads,
        mode=BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
        image_width=image_width,
        image_height=image_height,
    )
    pixel_score = score_bbox_candidate_payloads(image_path, pixel_payloads)
    normalized_score = score_bbox_candidate_payloads(image_path, normalized_payloads)
    scores = {
        BBOX_MODE_PIXEL_AS_IS: pixel_score,
        BBOX_MODE_NORMALIZED_1000_TO_PIXEL: normalized_score,
    }
    preferred_mode = (
        BBOX_MODE_NORMALIZED_1000_TO_PIXEL
        if normalized_score > (pixel_score + BBOX_MODE_SWITCH_MARGIN)
        else BBOX_MODE_PIXEL_AS_IS
    )
    page_gap = abs(normalized_score - pixel_score)
    image = QImage(str(image_path))
    if image.isNull():
        if preferred_mode == BBOX_MODE_NORMALIZED_1000_TO_PIXEL:
            return BBoxResolutionResult(
                payloads=normalized_payloads,
                mode=BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
                scores=scores,
                fact_modes=[BBOX_MODE_NORMALIZED_1000_TO_PIXEL for _ in normalized_payloads],
                policy="page_locked",
            )
        return BBoxResolutionResult(
            payloads=pixel_payloads,
            mode=BBOX_MODE_PIXEL_AS_IS,
            scores=scores,
            fact_modes=[BBOX_MODE_PIXEL_AS_IS for _ in pixel_payloads],
            policy="page_locked",
        )
    if page_gap >= BBOX_PAGE_MODE_LOCK_MARGIN:
        if preferred_mode == BBOX_MODE_NORMALIZED_1000_TO_PIXEL:
            return BBoxResolutionResult(
                payloads=normalized_payloads,
                mode=BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
                scores=scores,
                fact_modes=[BBOX_MODE_NORMALIZED_1000_TO_PIXEL for _ in normalized_payloads],
                policy="page_locked",
            )
        return BBoxResolutionResult(
            payloads=pixel_payloads,
            mode=BBOX_MODE_PIXEL_AS_IS,
            scores=scores,
            fact_modes=[BBOX_MODE_PIXEL_AS_IS for _ in pixel_payloads],
            policy="page_locked",
        )

    resolved_payloads: list[dict[str, Any]] = []
    chosen_modes: set[str] = set()
    fact_modes: list[str] = []
    for pixel_payload, normalized_payload in zip(pixel_payloads, normalized_payloads):
        pixel_fact_score, _pixel_coverage = score_bbox_payload_ink(image, pixel_payload)
        normalized_fact_score, _normalized_coverage = score_bbox_payload_ink(image, normalized_payload)
        if preferred_mode == BBOX_MODE_NORMALIZED_1000_TO_PIXEL:
            choose_normalized = not (pixel_fact_score > (normalized_fact_score + BBOX_MODE_SWITCH_MARGIN))
        else:
            choose_normalized = normalized_fact_score > (pixel_fact_score + BBOX_MODE_SWITCH_MARGIN)
        if choose_normalized:
            resolved_payloads.append(deepcopy(normalized_payload))
            chosen_modes.add(BBOX_MODE_NORMALIZED_1000_TO_PIXEL)
            fact_modes.append(BBOX_MODE_NORMALIZED_1000_TO_PIXEL)
        else:
            resolved_payloads.append(deepcopy(pixel_payload))
            chosen_modes.add(BBOX_MODE_PIXEL_AS_IS)
            fact_modes.append(BBOX_MODE_PIXEL_AS_IS)

    if chosen_modes == {BBOX_MODE_NORMALIZED_1000_TO_PIXEL}:
        resolved_mode = BBOX_MODE_NORMALIZED_1000_TO_PIXEL
    elif chosen_modes == {BBOX_MODE_PIXEL_AS_IS}:
        resolved_mode = BBOX_MODE_PIXEL_AS_IS
    else:
        resolved_mode = BBOX_MODE_MIXED_AUTO
    return BBoxResolutionResult(
        payloads=resolved_payloads,
        mode=resolved_mode,
        scores=scores,
        fact_modes=fact_modes,
        policy="mixed_auto",
    )


def ordered_fact_payloads_by_geometry(
    fact_payloads: list[dict[str, Any]],
    *,
    reading_direction: str,
) -> list[dict[str, Any]]:
    if len(fact_payloads) <= 1:
        return [deepcopy(payload) for payload in fact_payloads]
    facts_for_order = [{"bbox": normalize_bbox_data(payload.get("bbox"))} for payload in fact_payloads]
    ordered_indices = canonical_fact_order_indices(
        facts_for_order,
        direction="rtl" if reading_direction == "rtl" else "ltr",
        row_tolerance_ratio=0.35,
        row_tolerance_min_px=6.0,
    )
    return [deepcopy(fact_payloads[idx]) for idx in ordered_indices if 0 <= idx < len(fact_payloads)]


__all__ = [
    "BBOX_MODE_NORMALIZED_1000_TO_PIXEL",
    "BBOX_MODE_PIXEL_AS_IS",
    "BBOX_MODE_MIXED_AUTO",
    "BBoxResolutionResult",
    "bbox_looks_normalized_1000",
    "normalize_ai_fact_payload",
    "normalize_bbox_to_list",
    "ordered_fact_payloads_by_geometry",
    "payloads_for_bbox_mode",
    "resolve_bbox_mode",
    "score_bbox_candidate_payloads",
    "score_bbox_payload_ink",
]
