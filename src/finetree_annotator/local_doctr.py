from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

LOCAL_DETECTOR_MODEL_NAME = "Fine-Tuned Detector"
LOCAL_DOCTR_MODEL_NAME = LOCAL_DETECTOR_MODEL_NAME
STOCK_DOCTR_DETECTOR_MODEL_NAME = "docTR Pretrained Detector"
MERGED_DETECTOR_MODEL_NAME = "Merged Detector (stock + fine-tuned)"
DETECTOR_MODEL_ENV_VAR = "FINETREE_DETECTOR_MODEL_PATH"
DETECTOR_MODEL_FILENAME = "db_resnet50_20260328-203923.pt"
DEFAULT_DETECTOR_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "models" / "detection" / DETECTOR_MODEL_FILENAME
)
DEFAULT_DOCTR_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "doctr_cache"
DEFAULT_DOCTR_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "doctr_logs"
LOCAL_DETECTOR_BACKEND_FINE_TUNED = "fine_tuned"
LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED = "doctr_pretrained"
LOCAL_DETECTOR_BACKEND_MERGED = "merged"
SOURCE_FINE_TUNED = "fine_tuned"
SOURCE_OLD = "old"
MATCH_INTERSECTION_OVER_OLD_THRESHOLD = 0.75
MATCH_IOU_THRESHOLD = 0.25
FINE_TUNED_BBOX_SCALE = 0.33

NUMERIC_TOKEN_RE = re.compile(
    r"""^
        -$ |
        [\-\+\(]?
        (?:[$€£₪])?
        (?:
            \d{1,3}(?:,\d{3})+ |
            \d+
        )
        (?:\.\d+)?
        %?
        [\)]?
        $
    """,
    re.VERBOSE,
)


def resolve_detector_model_path() -> Path:
    override = os.environ.get(DETECTOR_MODEL_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_DETECTOR_MODEL_PATH


def normalize_local_detector_backend(backend: Any) -> str:
    value = str(getattr(backend, "value", backend) or "").strip().lower()
    if value == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
        return LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED
    if value == LOCAL_DETECTOR_BACKEND_MERGED:
        return LOCAL_DETECTOR_BACKEND_MERGED
    return LOCAL_DETECTOR_BACKEND_FINE_TUNED


def local_detector_model_name(backend: Any) -> str:
    normalized = normalize_local_detector_backend(backend)
    if normalized == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
        return STOCK_DOCTR_DETECTOR_MODEL_NAME
    if normalized == LOCAL_DETECTOR_BACKEND_MERGED:
        return MERGED_DETECTOR_MODEL_NAME
    return LOCAL_DETECTOR_MODEL_NAME


def _ensure_doctr_cache_dir() -> None:
    if not os.environ.get("DOCTR_CACHE_DIR", "").strip():
        os.environ["DOCTR_CACHE_DIR"] = str(DEFAULT_DOCTR_CACHE_DIR)


def is_excluded_numeric_token_text(value: Any) -> bool:
    text = str(value or "").strip()
    if text == "31":
        return True
    if len(text) == 4 and text.isdigit():
        year = int(text)
        if 2000 <= year <= 2026:
            return True
    return False


def is_numeric_token_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not NUMERIC_TOKEN_RE.match(text):
        return False
    return not is_excluded_numeric_token_text(text)


def doctr_geometry_to_bbox(
    geometry: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    if not isinstance(geometry, (list, tuple)) or len(geometry) != 2:
        return None
    top_left, bottom_right = geometry
    if not isinstance(top_left, (list, tuple)) or not isinstance(bottom_right, (list, tuple)):
        return None
    if len(top_left) != 2 or len(bottom_right) != 2:
        return None
    try:
        xmin = float(top_left[0])
        ymin = float(top_left[1])
        xmax = float(bottom_right[0])
        ymax = float(bottom_right[1])
    except (TypeError, ValueError):
        return None

    left = max(0, min(int(image_width), int(xmin * image_width)))
    top = max(0, min(int(image_height), int(ymin * image_height)))
    right = max(0, min(int(image_width), int(xmax * image_width)))
    bottom = max(0, min(int(image_height), int(ymax * image_height)))
    width = max(0, right - left)
    height = max(0, bottom - top)
    if width <= 0 or height <= 0:
        return None
    return [left, top, width, height]


def detector_word_to_bbox(
    raw_box: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    try:
        xmin = float(raw_box[0])
        ymin = float(raw_box[1])
        xmax = float(raw_box[2])
        ymax = float(raw_box[3])
    except (IndexError, TypeError, ValueError):
        return None
    return doctr_geometry_to_bbox(
        ((xmin, ymin), (xmax, ymax)),
        image_width=image_width,
        image_height=image_height,
    )


def normalize_fine_tuned_bbox(
    bbox: list[int],
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    scaled_w = max(1, int(round(w * FINE_TUNED_BBOX_SCALE)))
    scaled_h = max(1, int(round(h * FINE_TUNED_BBOX_SCALE)))
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    left = int(round(center_x - (scaled_w / 2.0)))
    top = int(round(center_y - (scaled_h / 2.0)))
    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    scaled_w = max(1, min(scaled_w, image_width - left))
    scaled_h = max(1, min(scaled_h, image_height - top))
    return [left, top, scaled_w, scaled_h]


def detector_checkpoint_missing_message(model_path: Path) -> str:
    return (
        "Fine-tuned detector checkpoint not found.\n\n"
        f"Resolved checkpoint path: {model_path}\n"
        f"Default checkpoint path: {DEFAULT_DETECTOR_MODEL_PATH}\n"
        f"Override with {DETECTOR_MODEL_ENV_VAR}=/absolute/path/to/{DETECTOR_MODEL_FILENAME}"
    )


def doctr_unavailable_message(exc: Exception | None = None) -> str:
    detail = f"\n\nOriginal error:\n{exc}" if exc is not None else ""
    return (
        "Local bbox detector is unavailable. Install `python-doctr[torch]`, `torch`, `numpy`, and `Pillow` "
        "in the environment that runs FineTree."
        f"{detail}"
    )


def pretrained_detector_unavailable_message(exc: Exception | None = None) -> str:
    detail = f"\n\nOriginal error:\n{exc}" if exc is not None else ""
    return (
        "Stock docTR detector is unavailable. On first use, docTR may need to download pretrained weights into:\n"
        f"{DEFAULT_DOCTR_CACHE_DIR}\n\n"
        "Ensure the runtime can access the network once, or pre-populate that cache directory."
        f"{detail}"
    )


def _resolve_state_dict(payload: Any) -> Any:
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
    return payload


@lru_cache(maxsize=4)
def _load_fine_tuned_detection_model(model_path_str: str) -> Any:
    model_path = Path(model_path_str)
    if not model_path.is_file():
        raise FileNotFoundError(detector_checkpoint_missing_message(model_path))
    try:
        import torch
        from doctr.models import db_resnet50, detection_predictor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(doctr_unavailable_message(exc)) from exc

    model = db_resnet50(pretrained=False)
    state_dict = _resolve_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.load_state_dict(state_dict, strict=True)
    return detection_predictor(arch=model, pretrained=False, assume_straight_pages=True)


@lru_cache(maxsize=1)
def _load_pretrained_detection_model() -> Any:
    _ensure_doctr_cache_dir()
    try:
        from doctr.models import detection_predictor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(doctr_unavailable_message(exc)) from exc
    try:
        return detection_predictor(
            arch="db_resnet50",
            pretrained=True,
            assume_straight_pages=True,
        )
    except Exception as exc:
        raise RuntimeError(pretrained_detector_unavailable_message(exc)) from exc


def _load_detector_for_backend(backend: Any) -> Any:
    normalized_backend = normalize_local_detector_backend(backend)
    if normalized_backend == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
        return _load_pretrained_detection_model()
    model_path = resolve_detector_model_path()
    return _load_fine_tuned_detection_model(str(model_path))


@lru_cache(maxsize=1)
def _load_recognition_model() -> Any:
    try:
        from doctr.models import recognition_predictor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(doctr_unavailable_message(exc)) from exc
    return recognition_predictor(pretrained=True)


def _recognition_output_to_result(payload: Any) -> tuple[str, float | None]:
    text = ""
    confidence: float | None = None
    if isinstance(payload, (list, tuple)) and payload:
        text = str(payload[0] or "").strip()
        if len(payload) > 1:
            try:
                confidence = float(payload[1])
            except (TypeError, ValueError):
                confidence = None
        return text, confidence
    return str(payload or "").strip(), None


def _recognize_crop_payloads(
    crops: list[Any],
    *,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> list[tuple[str, float | None]]:
    if not crops:
        return []

    recognizer = _load_recognition_model()
    try:
        outputs = recognizer(crops)
    except Exception:
        outputs = None

    if isinstance(outputs, list) and len(outputs) == len(crops):
        return [_recognition_output_to_result(payload) for payload in outputs]

    results: list[tuple[str, float | None]] = []
    for crop in crops:
        if cancel_requested is not None and cancel_requested():
            return results
        try:
            payloads = recognizer([crop])
        except Exception:
            results.append(("", None))
            continue
        payload = payloads[0] if isinstance(payloads, list) and payloads else ""
        results.append(_recognition_output_to_result(payload))
    return results


def _collect_detection_candidates(
    detected_words: Any,
    *,
    image_width: int,
    image_height: int,
    source: str,
) -> list[dict[str, Any]]:
    rows = getattr(detected_words, "tolist", lambda: detected_words)()
    if not isinstance(rows, list):
        return []

    candidates: list[dict[str, Any]] = []
    for raw_box in rows:
        bbox = detector_word_to_bbox(
            raw_box,
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            continue
        score = 0.0
        try:
            score = float(raw_box[4])
        except (IndexError, TypeError, ValueError):
            score = 0.0
        candidates.append(
            {
                "bbox": bbox,
                "score": score,
                "value": "",
                "confidence": None,
                "source": source,
            }
        )
    candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["score"]))
    return candidates


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _bbox_intersection_area(first: list[int], second: list[int]) -> int:
    first_right = first[0] + first[2]
    first_bottom = first[1] + first[3]
    second_right = second[0] + second[2]
    second_bottom = second[1] + second[3]
    overlap_w = max(0, min(first_right, second_right) - max(first[0], second[0]))
    overlap_h = max(0, min(first_bottom, second_bottom) - max(first[1], second[1]))
    return overlap_w * overlap_h


def _iou(first: list[int], second: list[int]) -> float:
    intersection = _bbox_intersection_area(first, second)
    if intersection <= 0:
        return 0.0
    first_area = _bbox_area(first)
    second_area = _bbox_area(second)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _intersection_over_old(old_bbox: list[int], other_bbox: list[int]) -> float:
    old_area = _bbox_area(old_bbox)
    if old_area <= 0:
        return 0.0
    return _bbox_intersection_area(old_bbox, other_bbox) / old_area


def _candidate_value(candidate: dict[str, Any]) -> str:
    return str(candidate.get("value") or "").strip()


def _candidate_confidence(candidate: dict[str, Any]) -> float | None:
    raw = candidate.get("confidence")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_matched_value(old_candidate: dict[str, Any], fine_candidate: dict[str, Any]) -> str:
    old_value = _candidate_value(old_candidate)
    fine_value = _candidate_value(fine_candidate)
    if not old_value and fine_value:
        return fine_value
    if old_value and not fine_value:
        return old_value
    if fine_value == "-" and old_value != "-":
        return fine_value
    if old_value == fine_value:
        return old_value
    if not old_value and not fine_value:
        return ""

    old_conf = _candidate_confidence(old_candidate)
    fine_conf = _candidate_confidence(fine_candidate)
    if old_conf is not None and fine_conf is not None and old_conf > fine_conf + 0.05:
        return old_value
    return fine_value or old_value


def _limit_results(items: list[dict[str, Any]], *, max_facts: int) -> list[dict[str, Any]]:
    if max_facts > 0:
        return items[:max_facts]
    return items


def _write_doctr_debug_log(image_path: Path, payload: dict[str, Any]) -> Path | None:
    try:
        DEFAULT_DOCTR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        log_path = DEFAULT_DOCTR_LOG_DIR / f"{timestamp}_{image_path.stem}.json"
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return log_path
    except Exception:
        return None


def extract_numeric_bbox_candidates(
    image_path: Path,
    *,
    cancel_requested: Optional[Callable[[], bool]] = None,
    backend: Any = LOCAL_DETECTOR_BACKEND_FINE_TUNED,
) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(doctr_unavailable_message(exc)) from exc

    if cancel_requested is not None and cancel_requested():
        return []

    normalized_backend = normalize_local_detector_backend(backend)
    if normalized_backend == LOCAL_DETECTOR_BACKEND_MERGED:
        raise ValueError("Merged backend is not supported for candidate extraction.")

    detector = _load_detector_for_backend(normalized_backend)
    source = SOURCE_OLD if normalized_backend == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED else SOURCE_FINE_TUNED

    with Image.open(image_path) as src_img:
        rgb_img = src_img.convert("RGB")
        image_width, image_height = rgb_img.size
        detection_result = detector([np.asarray(rgb_img)])

        if cancel_requested is not None and cancel_requested():
            return []

        page_result = detection_result[0] if isinstance(detection_result, list) and detection_result else {}
        candidates = _collect_detection_candidates(
            page_result.get("words"),
            image_width=image_width,
            image_height=image_height,
            source=source,
        )
        if not candidates:
            return []

        crop_arrays: list[Any] = []
        crop_candidate_indices: list[int] = []
        recognition_results: list[tuple[str, float | None]] = [("", None)] * len(candidates)
        for index, candidate in enumerate(candidates):
            if cancel_requested is not None and cancel_requested():
                return []
            x, y, w, h = candidate["bbox"]
            crop_array = np.asarray(rgb_img.crop((x, y, x + w, y + h)))
            if crop_array.ndim != 3 or crop_array.shape[0] <= 0 or crop_array.shape[1] <= 0:
                continue
            crop_arrays.append(crop_array)
            crop_candidate_indices.append(index)

    crop_results = _recognize_crop_payloads(crop_arrays, cancel_requested=cancel_requested)
    for index, result in zip(crop_candidate_indices, crop_results):
        recognition_results[index] = result

    filtered: list[dict[str, Any]] = []
    for candidate, (text, confidence) in zip(candidates, recognition_results):
        if cancel_requested is not None and cancel_requested():
            return filtered
        normalized_text = str(text or "").strip()
        if normalized_text and not is_numeric_token_text(normalized_text):
            continue
        enriched = dict(candidate)
        if enriched.get("source") == SOURCE_FINE_TUNED and normalized_text == "-":
            normalized_bbox = normalize_fine_tuned_bbox(
                list(enriched["bbox"]),
                image_width=image_width,
                image_height=image_height,
            )
            if normalized_bbox is not None:
                enriched["bbox"] = normalized_bbox
        enriched["value"] = normalized_text
        enriched["confidence"] = confidence
        filtered.append(enriched)
    return filtered


def _match_old_and_fine_candidates(
    old_candidates: list[dict[str, Any]],
    fine_candidates: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    scored_pairs: list[tuple[float, float, int, int]] = []
    for old_index, old_candidate in enumerate(old_candidates):
        old_bbox = list(old_candidate["bbox"])
        for fine_index, fine_candidate in enumerate(fine_candidates):
            fine_bbox = list(fine_candidate["bbox"])
            intersection_over_old = _intersection_over_old(old_bbox, fine_bbox)
            iou = _iou(old_bbox, fine_bbox)
            if intersection_over_old >= MATCH_INTERSECTION_OVER_OLD_THRESHOLD or iou >= MATCH_IOU_THRESHOLD:
                scored_pairs.append((intersection_over_old, iou, old_index, fine_index))

    scored_pairs.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    matched_old: set[int] = set()
    matched_fine: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _intersection, _iou_score, old_index, fine_index in scored_pairs:
        if old_index in matched_old or fine_index in matched_fine:
            continue
        matched_old.add(old_index)
        matched_fine.add(fine_index)
        matches.append((old_index, fine_index))
    return matches


def _merge_detector_candidates(
    old_candidates: list[dict[str, Any]],
    fine_candidates: list[dict[str, Any]],
    *,
    cancel_requested: Optional[Callable[[], bool]] = None,
    max_facts: int = 0,
) -> list[dict[str, Any]]:
    matches = _match_old_and_fine_candidates(old_candidates, fine_candidates)
    matched_old = {old_index for old_index, _fine_index in matches}
    matched_fine = {fine_index for _old_index, fine_index in matches}

    merged: list[dict[str, Any]] = []
    for old_index, fine_index in matches:
        if cancel_requested is not None and cancel_requested():
            return _limit_results(merged, max_facts=max_facts)
        old_candidate = old_candidates[old_index]
        fine_candidate = fine_candidates[fine_index]
        merged.append(
            {
                "bbox": list(old_candidate["bbox"]),
                "value": _resolve_matched_value(old_candidate, fine_candidate),
                "source": "matched",
            }
        )

    for fine_index, fine_candidate in enumerate(fine_candidates):
        if cancel_requested is not None and cancel_requested():
            return _limit_results(merged, max_facts=max_facts)
        if fine_index in matched_fine:
            continue
        merged.append(
            {
                "bbox": list(fine_candidate["bbox"]),
                "value": _candidate_value(fine_candidate),
                "source": SOURCE_FINE_TUNED,
            }
        )

    merged.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item.get("source", "")))
    return _limit_results(merged, max_facts=max_facts)


def extract_numeric_bbox_facts(
    image_path: Path,
    *,
    max_facts: int = 0,
    cancel_requested: Optional[Callable[[], bool]] = None,
    backend: Any = LOCAL_DETECTOR_BACKEND_MERGED,
) -> list[dict[str, Any]]:
    normalized_backend = normalize_local_detector_backend(backend)
    if normalized_backend == LOCAL_DETECTOR_BACKEND_MERGED:
        old_candidates = extract_numeric_bbox_candidates(
            image_path,
            cancel_requested=cancel_requested,
            backend=LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED,
        )
        if cancel_requested is not None and cancel_requested():
            return []
        fine_candidates = extract_numeric_bbox_candidates(
            image_path,
            cancel_requested=cancel_requested,
            backend=LOCAL_DETECTOR_BACKEND_FINE_TUNED,
        )
        merged = _merge_detector_candidates(
            old_candidates,
            fine_candidates,
            cancel_requested=cancel_requested,
            max_facts=max_facts,
        )
        facts = [{"bbox": item["bbox"], "value": item["value"]} for item in merged]
        _write_doctr_debug_log(
            image_path,
            {
                "backend": normalized_backend,
                "image_path": str(image_path),
                "old_candidates": old_candidates,
                "fine_candidates": fine_candidates,
                "merged_candidates": merged,
                "facts": facts,
            },
        )
        return facts

    candidates = extract_numeric_bbox_candidates(
        image_path,
        cancel_requested=cancel_requested,
        backend=normalized_backend,
    )
    candidates = _limit_results(candidates, max_facts=max_facts)
    facts = [{"bbox": list(item["bbox"]), "value": _candidate_value(item)} for item in candidates]
    _write_doctr_debug_log(
        image_path,
        {
            "backend": normalized_backend,
            "image_path": str(image_path),
            "candidates": candidates,
            "facts": facts,
        },
    )
    return facts


__all__ = [
    "DEFAULT_DETECTOR_MODEL_PATH",
    "DEFAULT_DOCTR_LOG_DIR",
    "DEFAULT_DOCTR_CACHE_DIR",
    "DETECTOR_MODEL_ENV_VAR",
    "DETECTOR_MODEL_FILENAME",
    "LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED",
    "LOCAL_DETECTOR_BACKEND_FINE_TUNED",
    "LOCAL_DETECTOR_BACKEND_MERGED",
    "LOCAL_DETECTOR_MODEL_NAME",
    "LOCAL_DOCTR_MODEL_NAME",
    "MATCH_INTERSECTION_OVER_OLD_THRESHOLD",
    "MATCH_IOU_THRESHOLD",
    "MERGED_DETECTOR_MODEL_NAME",
    "NUMERIC_TOKEN_RE",
    "STOCK_DOCTR_DETECTOR_MODEL_NAME",
    "SOURCE_FINE_TUNED",
    "SOURCE_OLD",
    "detector_checkpoint_missing_message",
    "detector_word_to_bbox",
    "doctr_geometry_to_bbox",
    "doctr_unavailable_message",
    "extract_numeric_bbox_candidates",
    "extract_numeric_bbox_facts",
    "FINE_TUNED_BBOX_SCALE",
    "is_excluded_numeric_token_text",
    "is_numeric_token_text",
    "local_detector_model_name",
    "normalize_fine_tuned_bbox",
    "normalize_local_detector_backend",
    "pretrained_detector_unavailable_message",
    "resolve_detector_model_path",
]
