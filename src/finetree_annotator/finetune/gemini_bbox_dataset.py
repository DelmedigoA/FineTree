from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from datasets import DatasetDict
from .bbox_even_quantization import quantize_yxyx_bbox_to_even
from .config import load_finetune_config
from .dataset_builder import (
    _doc_split_map,
    _iter_annotation_files,
    _resolve_page_image_path,
    _transform_page_for_target,
)
from .duplicate_facts import assert_no_duplicate_facts
from .push_dataset_hub import (
    _copy_or_resize_image,
    _parse_doc_ids_csv,
    _resolve_resize_bounds,
    _resolve_system_prompt,
    build_hf_dataset_from_export,
    push_to_hf,
    resolve_hf_token,
)
from ..ai.payloads import build_extraction_prompt
from ..fact_normalization import assert_fact_format
from ..fact_ordering import assert_fact_order, normalize_document_meta, resolve_reading_direction
from ..schema_contract import PROMPT_FACT_KEYS, PROMPT_PAGE_META_KEYS, build_gemini_bbox_page_schema_preview
from ..schema_io import load_any_schema

InstructionMode = Literal["source", "minimal"]
GeminiBBoxSourceFormat = Literal[
    "pixel_xywh",
    "normalized_1000_xywh",
    "normalized_1000_yxyx",
    "auto",
]

DEFAULT_CONFIG_PATH = Path("configs/finetune_qwen35a3_vl.yaml")
DEFAULT_EXPORT_DIR = Path("artifacts/hf_gemini_bbox_dataset")
DEFAULT_SOURCE_BBOX_FORMAT: GeminiBBoxSourceFormat = "pixel_xywh"
MINIMAL_GEMINI_BBOX_INSTRUCTION = (
    "Extract the FineTree page-level JSON. Use `bbox` as [ymin, xmin, ymax, xmax] "
    "normalized to 0-1000 integers relative to the provided image."
)
_MIN_PIXEL_SIZE = 1e-3
_EPS = 1e-6


@dataclass
class GeminiBBoxExportStats:
    annotation_files: int = 0
    pages_seen: int = 0
    train_rows: int = 0
    val_rows: int = 0
    pages_skipped_empty: int = 0
    pages_skipped_missing_image: int = 0
    pages_skipped_unapproved: int = 0
    resized_images: int = 0
    unchanged_images: int = 0
    bbox_converted: int = 0
    bbox_clamped: int = 0
    skipped_rows_by_include_filter: int = 0
    preview_images_created: int = 0
    facts_deduped: int = 0


def _normalize_instruction_mode(value: str) -> InstructionMode:
    mode = str(value or "").strip().lower()
    if mode not in {"source", "minimal"}:
        raise ValueError("--instruction-mode must be one of: source, minimal")
    return mode  # type: ignore[return-value]


def _normalize_source_bbox_format(value: str) -> GeminiBBoxSourceFormat:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {
        "pixel_xywh",
        "normalized_1000_xywh",
        "normalized_1000_yxyx",
        "auto",
    }:
        raise ValueError(
            "--source-bbox-format must be one of: pixel_xywh, normalized_1000_xywh, "
            "normalized_1000_yxyx, auto"
        )
    return normalized  # type: ignore[return-value]


def build_gemini_bbox_prompt_template(
    *,
    page_meta_keys: Sequence[str] | None = None,
    fact_keys: Sequence[str] | None = None,
) -> str:
    schema_preview = build_gemini_bbox_page_schema_preview(
        page_meta_keys=tuple(page_meta_keys or PROMPT_PAGE_META_KEYS),
        fact_keys=tuple(fact_keys or PROMPT_FACT_KEYS),
    )
    return "\n".join(
        [
            "You are extracting financial-statement annotations from a single page image.",
            "",
            "Current image size: {{IMAGE_DIMENSIONS}}.",
            "",
            "Return ONLY valid JSON.",
            "Do NOT return markdown, code fences, comments, prose, or extra keys.",
            "",
            "Return the exact page-level object shown below.",
            "",
            "Exact schema:",
            schema_preview,
            "",
            "1. `bbox` must be `[ymin, xmin, ymax, xmax]`.",
            "2. Each bbox coordinate must be an integer normalized to `0-1000` relative to the provided image.",
            "3. `ymin < ymax` and `xmin < xmax` must always hold.",
            "4. Return only a single page-level object with `meta` and `facts`.",
            "5. Include all listed `meta` keys and all listed fact keys in every emitted fact. Use JSON `null` for missing optional values.",
            "6. Extract only visible numeric or numeric-symbol facts. Do not emit standalone labels, headings, row labels, or captions as facts.",
            "7. Preserve value text exactly as printed, including `%`, commas, parentheses, and dash placeholders.",
            "8. `fact_num` must be contiguous integers starting at 1 and must match the emitted fact order.",
            "9. Order facts top-to-bottom; within a row use right-to-left for Hebrew pages and left-to-right for English pages.",
            "10. Keep `path` as visible hierarchy labels only; use `[]` when unknown.",
            "11. If `comment_ref` seems unreasonably long, do not include the full text. Use a short marker only, for example `\"*\"` or `\"(1)\"`.",
            "12. `equations` must be a JSON list or `null`. Use `null` unless the page visibly supports a reliable arithmetic relation.",
            "13. Extract a fact only when you can localize its numeric cell tightly. Skip uncertain or non-localizable facts.",
            "14. Keep UTF-8 Hebrew directly; do not escape it to unicode sequences.",
        ]
    ).strip()


def _iter_fact_dicts(payload: Any) -> list[dict[str, Any]]:
    facts_out: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return facts_out
    facts = payload.get("facts")
    if isinstance(facts, list):
        facts_out.extend([fact for fact in facts if isinstance(fact, dict)])
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_facts = page.get("facts")
            if isinstance(page_facts, list):
                facts_out.extend([fact for fact in page_facts if isinstance(fact, dict)])
    return facts_out


def _parse_bbox_quad(raw_bbox: Any) -> tuple[float, float, float, float]:
    if isinstance(raw_bbox, dict):
        values = (
            raw_bbox.get("x"),
            raw_bbox.get("y"),
            raw_bbox.get("w"),
            raw_bbox.get("h"),
        )
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        values = (raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3])
    else:
        raise ValueError(f"Unsupported bbox shape: {raw_bbox!r}")
    try:
        return tuple(float(value) for value in values)  # type: ignore[return-value]
    except Exception as exc:
        raise ValueError(f"BBox values must be numeric: {raw_bbox!r}") from exc


def _pixel_xywh_plausible(
    bbox: tuple[float, float, float, float],
    *,
    image_width: float,
    image_height: float,
) -> bool:
    x, y, w, h = bbox
    return (
        x >= -_EPS
        and y >= -_EPS
        and w > 0.0
        and h > 0.0
        and (x + w) <= (float(image_width) + _EPS)
        and (y + h) <= (float(image_height) + _EPS)
    )


def _normalized_xywh_plausible(bbox: tuple[float, float, float, float]) -> bool:
    x, y, w, h = bbox
    return (
        x >= -_EPS
        and y >= -_EPS
        and w > 0.0
        and h > 0.0
        and x <= (1000.0 + _EPS)
        and y <= (1000.0 + _EPS)
        and w <= (1000.0 + _EPS)
        and h <= (1000.0 + _EPS)
        and (x + w) <= (1000.0 + _EPS)
        and (y + h) <= (1000.0 + _EPS)
    )


def _normalized_yxyx_plausible(bbox: tuple[float, float, float, float]) -> bool:
    ymin, xmin, ymax, xmax = bbox
    return (
        ymin >= -_EPS
        and xmin >= -_EPS
        and ymax <= (1000.0 + _EPS)
        and xmax <= (1000.0 + _EPS)
        and ymin < ymax
        and xmin < xmax
    )


def _resolve_detected_bbox_source_format(
    raw_bbox: Any,
    *,
    image_width: float,
    image_height: float,
    source_format: GeminiBBoxSourceFormat,
) -> GeminiBBoxSourceFormat:
    normalized = _normalize_source_bbox_format(source_format)
    if normalized != "auto":
        return normalized

    parsed = _parse_bbox_quad(raw_bbox)
    candidates: list[GeminiBBoxSourceFormat] = []
    if _pixel_xywh_plausible(parsed, image_width=image_width, image_height=image_height):
        candidates.append("pixel_xywh")
    if _normalized_xywh_plausible(parsed):
        candidates.append("normalized_1000_xywh")
    if _normalized_yxyx_plausible(parsed):
        candidates.append("normalized_1000_yxyx")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"Could not infer bbox source format for {raw_bbox!r} against image size "
            f"{int(image_width)}x{int(image_height)}."
        )
    raise ValueError(
        f"Ambiguous bbox source format for {raw_bbox!r}: plausible formats are {', '.join(candidates)}. "
        "Pass --source-bbox-format explicitly."
    )


def _source_bbox_to_pixel_xywh(
    raw_bbox: Any,
    *,
    image_width: float,
    image_height: float,
    source_format: GeminiBBoxSourceFormat,
) -> tuple[float, float, float, float, GeminiBBoxSourceFormat]:
    resolved_format = _resolve_detected_bbox_source_format(
        raw_bbox,
        image_width=image_width,
        image_height=image_height,
        source_format=source_format,
    )
    a, b, c, d = _parse_bbox_quad(raw_bbox)
    if resolved_format == "pixel_xywh":
        if not _pixel_xywh_plausible((a, b, c, d), image_width=image_width, image_height=image_height):
            raise ValueError(
                f"Pixel xywh bbox {raw_bbox!r} exceeds image bounds "
                f"{int(image_width)}x{int(image_height)}."
            )
        return a, b, c, d, resolved_format
    if resolved_format == "normalized_1000_xywh":
        if not _normalized_xywh_plausible((a, b, c, d)):
            raise ValueError(f"Normalized xywh bbox {raw_bbox!r} is invalid.")
        return (
            (a / 1000.0) * float(image_width),
            (b / 1000.0) * float(image_height),
            (c / 1000.0) * float(image_width),
            (d / 1000.0) * float(image_height),
            resolved_format,
        )
    if resolved_format == "normalized_1000_yxyx":
        if not _normalized_yxyx_plausible((a, b, c, d)):
            raise ValueError(f"Normalized yxyx bbox {raw_bbox!r} is invalid.")
        return (
            (b / 1000.0) * float(image_width),
            (a / 1000.0) * float(image_height),
            ((d - b) / 1000.0) * float(image_width),
            ((c - a) / 1000.0) * float(image_height),
            resolved_format,
        )
    raise ValueError(f"Unsupported bbox source format: {resolved_format}")


def _clamp_pixel_xywh(
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float, bool]:
    clamped = False
    width = max(float(image_width), _MIN_PIXEL_SIZE)
    height = max(float(image_height), _MIN_PIXEL_SIZE)

    x0 = max(0.0, float(x))
    y0 = max(0.0, float(y))
    if x0 != float(x) or y0 != float(y):
        clamped = True

    max_x = max(width - _MIN_PIXEL_SIZE, 0.0)
    max_y = max(height - _MIN_PIXEL_SIZE, 0.0)
    x1 = min(x0, max_x)
    y1 = min(y0, max_y)
    if x1 != x0 or y1 != y0:
        clamped = True

    right = min(max(x1 + max(float(w), _MIN_PIXEL_SIZE), x1 + _MIN_PIXEL_SIZE), width)
    bottom = min(max(y1 + max(float(h), _MIN_PIXEL_SIZE), y1 + _MIN_PIXEL_SIZE), height)
    if right != (x1 + float(w)) or bottom != (y1 + float(h)):
        clamped = True

    w1 = max(right - x1, _MIN_PIXEL_SIZE)
    h1 = max(bottom - y1, _MIN_PIXEL_SIZE)
    return x1, y1, w1, h1, clamped


def validate_gemini_bbox_1000(bbox: Sequence[Any]) -> list[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Gemini bbox must be a 4-item sequence, got: {bbox!r}")
    values: list[int] = []
    for value in bbox:
        if not isinstance(value, int):
            raise ValueError(f"Gemini bbox coordinates must be integers, got: {bbox!r}")
        if value < 0 or value > 1000:
            raise ValueError(f"Gemini bbox coordinates must be in [0, 1000], got: {bbox!r}")
        if value % 2 != 0:
            raise ValueError(f"Gemini bbox coordinates must be even integers, got: {bbox!r}")
        values.append(int(value))
    ymin, xmin, ymax, xmax = values
    if ymin >= ymax:
        raise ValueError(f"Gemini bbox requires ymin < ymax, got: {bbox!r}")
    if xmin >= xmax:
        raise ValueError(f"Gemini bbox requires xmin < xmax, got: {bbox!r}")
    return values


def convert_bbox_to_gemini_1000(
    raw_bbox: Any,
    *,
    image_width: float,
    image_height: float,
    source_format: GeminiBBoxSourceFormat = DEFAULT_SOURCE_BBOX_FORMAT,
) -> tuple[list[int], bool, GeminiBBoxSourceFormat]:
    x, y, w, h, resolved_format = _source_bbox_to_pixel_xywh(
        raw_bbox,
        image_width=image_width,
        image_height=image_height,
        source_format=source_format,
    )
    x, y, w, h, clamped = _clamp_pixel_xywh(
        x=x,
        y=y,
        w=w,
        h=h,
        image_width=image_width,
        image_height=image_height,
    )
    xmin = (x / float(image_width)) * 1000.0
    ymin = (y / float(image_height)) * 1000.0
    xmax = ((x + w) / float(image_width)) * 1000.0
    ymax = ((y + h) / float(image_height)) * 1000.0
    bbox, was_adjusted = quantize_yxyx_bbox_to_even(
        ymin,
        xmin,
        ymax,
        xmax,
        upper_bound=1000,
    )
    bbox = validate_gemini_bbox_1000(bbox)
    return bbox, (clamped or was_adjusted), resolved_format


def _convert_payload_bboxes_to_gemini(
    payload: dict[str, Any],
    *,
    image_width: float,
    image_height: float,
    source_format: GeminiBBoxSourceFormat,
) -> tuple[dict[str, Any], int, int]:
    converted_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    converted = 0
    clamped = 0
    for fact in _iter_fact_dicts(converted_payload):
        if "bbox" not in fact:
            continue
        bbox, was_clamped, _resolved_format = convert_bbox_to_gemini_1000(
            fact.get("bbox"),
            image_width=image_width,
            image_height=image_height,
            source_format=source_format,
        )
        fact["bbox"] = bbox
        converted += 1
        if was_clamped:
            clamped += 1
    return converted_payload, converted, clamped


def _filter_gemini_fact_contexts(
    payload: dict[str, Any],
    *,
    allowed_value_contexts: set[str],
) -> tuple[dict[str, Any], int]:
    filtered_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    removed = 0
    facts = filtered_payload.get("facts")
    if not isinstance(facts, list):
        return filtered_payload, removed

    kept: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        value_context = str(fact.get("value_context") or "").strip().lower()
        if value_context not in allowed_value_contexts:
            removed += 1
            continue
        kept.append(fact)
    filtered_payload["facts"] = kept
    return filtered_payload, removed


def validate_gemini_payload(payload: dict[str, Any], *, source_label: str) -> int:
    validated = 0
    for fact in _iter_fact_dicts(payload):
        if "bbox" not in fact:
            continue
        fact["bbox"] = validate_gemini_bbox_1000(fact.get("bbox"))
        validated += 1
    return validated


def _write_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False))
    handle.write("\n")


def _iter_export_rows(export_dir: Path) -> list[tuple[str, int, dict[str, Any]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for split_name, path in (("train", export_dir / "train.jsonl"), ("validation", export_dir / "val.jsonl")):
        if not path.is_file():
            continue
        for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append((split_name, line_num, row))
    return rows


def _preview_pixel_bounds(
    bbox: Sequence[int],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = validate_gemini_bbox_1000(bbox)
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)

    x0 = max(0, min(width - 1, int(math.floor((xmin / 1000.0) * width))))
    y0 = max(0, min(height - 1, int(math.floor((ymin / 1000.0) * height))))
    x1 = max(x0 + 1, min(width, int(math.ceil((xmax / 1000.0) * width))))
    y1 = max(y0 + 1, min(height, int(math.ceil((ymax / 1000.0) * height))))
    return x0, y0, x1, y1


def create_gemini_bbox_preview_artifacts(export_dir: Path, *, count: int) -> dict[str, Any]:
    requested = max(int(count), 0)
    preview_dir = export_dir / "bbox_preview_samples"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if requested <= 0:
        summary = {"requested": 0, "created": 0, "output_dir": str(preview_dir), "samples": []}
        (preview_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to create Gemini bbox preview artifacts.") from exc

    created = 0
    samples: list[dict[str, Any]] = []
    for split_name, line_num, row in _iter_export_rows(export_dir):
        if created >= requested:
            break
        image_rel = row.get("image")
        text = row.get("text")
        if not isinstance(image_rel, str) or not image_rel.strip() or not isinstance(text, str) or not text.strip():
            continue

        image_path = (export_dir / image_rel).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing exported image for Gemini preview: {image_path}")
        payload = json.loads(text)

        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)

        drawn = 0
        for fact in _iter_fact_dicts(payload):
            raw_bbox = fact.get("bbox")
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                continue
            x0, y0, x1, y1 = _preview_pixel_bounds(
                [int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])],
                image_width=int(width),
                image_height=int(height),
            )
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 64, 64), width=2)
            drawn += 1

        if drawn <= 0:
            continue

        rel_image = Path(image_rel)
        doc_id = rel_image.parent.name or "unknown_doc"
        page_name = rel_image.stem or f"line_{line_num}"
        preview_name = f"{split_name}_{created + 1:02d}_{doc_id}_{page_name}.png"
        preview_path = preview_dir / preview_name
        canvas.save(preview_path)
        samples.append(
            {
                "split": split_name,
                "line": line_num,
                "image": image_rel,
                "preview_image": str(preview_path.relative_to(export_dir).as_posix()),
                "bbox_drawn": drawn,
                "image_size": [int(width), int(height)],
            }
        )
        created += 1

    summary = {
        "requested": requested,
        "created": created,
        "output_dir": str(preview_dir),
        "samples": samples,
    }
    (preview_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("GEMINI_BBOX_PREVIEW_ARTIFACTS:", json.dumps(summary, ensure_ascii=False))
    return summary


def _prepare_rows(
    *,
    root: Path,
    export_dir: Path,
    config_path: Path,
    instruction_mode: InstructionMode,
    source_bbox_format: GeminiBBoxSourceFormat,
    include_doc_ids: set[str] | None,
    validation_doc_ids: set[str] | None,
    approved_pages_only: bool,
    allow_format_issues: bool,
    allow_ordering_issues: bool,
    allow_duplicate_facts: bool,
    remove_duplicates: bool,
    min_pixels: int | None,
    max_pixels: int | None,
) -> GeminiBBoxExportStats:
    cfg = load_finetune_config(config_path)
    if cfg.data.fact_format_enforce:
        format_report = assert_fact_format(
            root,
            annotations_glob=cfg.data.annotations_glob,
            fail_on_issues=not allow_format_issues,
            include_doc_ids=include_doc_ids,
        )
        if allow_format_issues and int(format_report["facts_with_issues"]) > 0:
            print(
                "WARNING: continuing with format issues because --allow-format-issues was set. "
                f"facts_with_issues={format_report['facts_with_issues']}"
            )
    else:
        print("FACT_FORMAT_AUDIT: skipped (data.fact_format_enforce=false)")
    if cfg.data.fact_order_enforce:
        ordering_report = assert_fact_order(
            root,
            annotations_glob=cfg.data.annotations_glob,
            default_direction=cfg.data.fact_order_default_on_uncertain,
            row_tolerance_ratio=cfg.data.fact_order_row_tolerance_ratio,
            row_tolerance_min_px=cfg.data.fact_order_row_tolerance_min_px,
            fail_on_issues=not allow_ordering_issues,
            include_doc_ids=include_doc_ids,
        )
        if allow_ordering_issues and int(ordering_report["pages_with_order_issues"]) > 0:
            print(
                "WARNING: continuing with ordering issues because --allow-ordering-issues was set. "
                f"pages_with_order_issues={ordering_report['pages_with_order_issues']}"
            )
    else:
        print("FACT_ORDER_AUDIT: skipped (data.fact_order_enforce=false)")
    duplicate_report = assert_no_duplicate_facts(
        root,
        annotations_glob=cfg.data.annotations_glob,
        fail_on_duplicates=not (allow_duplicate_facts or remove_duplicates),
        include_doc_ids=include_doc_ids,
    )
    if remove_duplicates and int(duplicate_report["duplicate_rows"]) > 0:
        print(
            "INFO: duplicate facts detected in source annotations; exact duplicates will be removed during Gemini export. "
            f"duplicate_rows={duplicate_report['duplicate_rows']}"
        )
    elif allow_duplicate_facts and int(duplicate_report["duplicate_rows"]) > 0:
        print(
            "WARNING: continuing with duplicate facts because --allow-duplicate-facts was set. "
            f"duplicate_rows={duplicate_report['duplicate_rows']}"
        )

    stats = GeminiBBoxExportStats()
    prompt_template = build_gemini_bbox_prompt_template()
    system_prompt = _resolve_system_prompt(root)
    resize_bounds = _resolve_resize_bounds(min_pixels, max_pixels)
    resolved_min = resize_bounds[0] if resize_bounds else None
    resolved_max = resize_bounds[1] if resize_bounds else None

    annotation_files = list(_iter_annotation_files(cfg.data.annotations_glob))
    if include_doc_ids is not None:
        filtered_files = [path for path in annotation_files if path.stem in include_doc_ids]
        stats.skipped_rows_by_include_filter = max(len(annotation_files) - len(filtered_files), 0)
        annotation_files = filtered_files

    split_map = _doc_split_map(
        annotation_files,
        cfg.data.val_ratio,
        forced_val_doc_ids=(validation_doc_ids if validation_doc_ids is not None else set(cfg.data.val_doc_ids)),
        force_explicit_val_doc_ids=(validation_doc_ids is not None),
    )

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    images_root = export_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    train_path = export_dir / "train.jsonl"
    val_path = export_dir / "val.jsonl"
    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for annotation_path in annotation_files:
            stats.annotation_files += 1
            doc_id = annotation_path.stem
            is_val_doc = bool(split_map.get(doc_id, False))
            out_f = val_f if is_val_doc else train_f

            payload = load_any_schema(json.loads(annotation_path.read_text(encoding="utf-8")))
            metadata = normalize_document_meta(payload.get("metadata"))
            direction_info = resolve_reading_direction(
                metadata if isinstance(payload, dict) else None,
                payload=payload,
                default_direction=cfg.data.fact_order_default_on_uncertain,
            )
            direction = str(direction_info["direction"])
            images_dir = str(payload.get("images_dir") or "").strip()
            pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                stats.pages_seen += 1

                meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
                page_status = str(meta.get("annotation_status") or "").strip().lower()
                if approved_pages_only and page_status != "approved":
                    stats.pages_skipped_unapproved += 1
                    continue

                image_name = str(page.get("image") or "").strip()
                if not image_name:
                    stats.pages_skipped_missing_image += 1
                    continue

                source_image_path = _resolve_page_image_path(cfg, annotation_path, payload, image_name)
                if not source_image_path.is_file():
                    stats.pages_skipped_missing_image += 1
                    continue

                source_width, source_height = _image_dimensions(source_image_path)

                target_obj, deduped_count = _transform_page_for_target(
                    cfg,
                    images_dir=images_dir,
                    metadata=metadata,
                    page=page,
                    direction=direction,
                    drop_date=False,
                    selected_page_meta_keys=PROMPT_PAGE_META_KEYS,
                    selected_fact_keys=PROMPT_FACT_KEYS,
                    page_only_wrapper=True,
                    excluded_value_contexts=None,
                    dedupe_exact_facts=remove_duplicates,
                )
                stats.facts_deduped += int(deduped_count)
                filtered_target_obj, _filtered_out = _filter_gemini_fact_contexts(
                    target_obj,
                    allowed_value_contexts={"tabular", "mixed"},
                )

                converted_payload, converted_count, clamped_count = _convert_payload_bboxes_to_gemini(
                    filtered_target_obj,
                    image_width=float(source_width),
                    image_height=float(source_height),
                    source_format=source_bbox_format,
                )
                validate_gemini_payload(
                    converted_payload,
                    source_label=f"{annotation_path.name}:{image_name}",
                )

                rel_image_path = Path("images") / doc_id / source_image_path.name
                exported_image_path = export_dir / rel_image_path
                exported_image_path.parent.mkdir(parents=True, exist_ok=True)
                resize_stats = _copy_or_resize_image(
                    source_image_path,
                    exported_image_path,
                    min_pixels=resolved_min,
                    max_pixels=resolved_max,
                )
                if resize_stats["resized"]:
                    stats.resized_images += 1
                else:
                    stats.unchanged_images += 1

                prompt_text = (
                    MINIMAL_GEMINI_BBOX_INSTRUCTION
                    if instruction_mode == "minimal"
                    else build_extraction_prompt(
                        prompt_template,
                        exported_image_path,
                        image_dimensions=(float(resize_stats["new_w"]), float(resize_stats["new_h"])),
                    )
                )
                row = {
                    "image": rel_image_path.as_posix(),
                    "system": system_prompt,
                    "instruction": prompt_text,
                    "text": json.dumps(converted_payload, ensure_ascii=False),
                }
                _write_jsonl_line(out_f, row)
                if is_val_doc:
                    stats.val_rows += 1
                else:
                    stats.train_rows += 1
                stats.bbox_converted += converted_count
                stats.bbox_clamped += clamped_count

    readme = "\n".join(
        [
            "# FineTree Gemini BBox Dataset",
            "",
            "Generated from current repository annotations.",
            "",
            "- Assistant payload is page-level JSON with Gemini bbox coordinates.",
            "- `bbox` format: `[ymin, xmin, ymax, xmax]`.",
            "- `bbox` space: normalized integers in `[0, 1000]` relative to the provided image.",
            "- `bbox` quantization: outward-safe even integers.",
            f"- Instruction mode: {instruction_mode}",
            f"- Source bbox format: {source_bbox_format}",
            f"- Resize enabled: {'yes' if resize_bounds else 'no'}",
            f"- Min pixels: {resolved_min if resolved_min is not None else 'unset'}",
            f"- Max pixels: {resolved_max if resolved_max is not None else 'unset'}",
            f"- Train rows: {stats.train_rows}",
            f"- Validation rows: {stats.val_rows}",
            f"- BBox converted: {stats.bbox_converted}",
            f"- BBox clamped: {stats.bbox_clamped}",
            f"- Facts deduped during export: {stats.facts_deduped}",
            f"- Resized images: {stats.resized_images}",
            f"- Unchanged images: {stats.unchanged_images}",
        ]
    ).strip()
    (export_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "instruction_mode": instruction_mode,
                "source_bbox_format": source_bbox_format,
                "bbox_format": "[ymin, xmin, ymax, xmax]",
                "bbox_space": "normalized_1000_integer",
                "bbox_quantization": "even_outward_safe",
                "min_pixels": resolved_min,
                "max_pixels": resolved_max,
                "allowed_value_contexts": ["mixed", "tabular"],
                "remove_duplicates": remove_duplicates,
                "stats": asdict(stats),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return stats


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to inspect source image dimensions.") from exc
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    return int(width), int(height)


def validate_gemini_export_dir(export_dir: Path) -> dict[str, Any]:
    report = {
        "train_rows": 0,
        "validation_rows": 0,
        "bbox_validated": 0,
    }
    for split_name, path in (("train", export_dir / "train.jsonl"), ("validation", export_dir / "val.jsonl")):
        if not path.is_file():
            continue
        for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_num}: missing text payload.")
            payload = json.loads(text)
            validated = validate_gemini_payload(payload, source_label=f"{path}:{line_num}")
            report["bbox_validated"] += validated
            if split_name == "train":
                report["train_rows"] += 1
            else:
                report["validation_rows"] += 1
    print("GEMINI_BBOX_EXPORT_VALIDATION:", json.dumps(report, ensure_ascii=False))
    return report


def build_hf_dataset_from_gemini_export(export_dir: Path) -> tuple[DatasetDict, int, int]:
    validate_gemini_export_dir(export_dir)
    return build_hf_dataset_from_export(export_dir)


def push_gemini_bbox_dataset_to_hf(
    dataset: DatasetDict,
    token: str,
    *,
    repo_id: str,
    private: bool = False,
) -> str:
    return push_to_hf(
        dataset,
        token,
        repo_id,
        private=private,
    )


def prepare_gemini_bbox_dataset(
    *,
    root: Path,
    config_path: Path,
    export_dir: Path,
    instruction_mode: InstructionMode = "source",
    source_bbox_format: GeminiBBoxSourceFormat = DEFAULT_SOURCE_BBOX_FORMAT,
    include_doc_ids: set[str] | None = None,
    validation_doc_ids: set[str] | None = None,
    approved_pages_only: bool = False,
    allow_format_issues: bool = False,
    allow_ordering_issues: bool = False,
    allow_duplicate_facts: bool = False,
    remove_duplicates: bool = False,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    preview_count: int = 6,
) -> GeminiBBoxExportStats:
    stats = _prepare_rows(
        root=root,
        export_dir=export_dir,
        config_path=config_path,
        instruction_mode=instruction_mode,
        source_bbox_format=source_bbox_format,
        include_doc_ids=include_doc_ids,
        validation_doc_ids=validation_doc_ids,
        approved_pages_only=approved_pages_only,
        allow_format_issues=allow_format_issues,
        allow_ordering_issues=allow_ordering_issues,
        allow_duplicate_facts=allow_duplicate_facts,
        remove_duplicates=remove_duplicates,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    validate_gemini_export_dir(export_dir)
    preview_summary = create_gemini_bbox_preview_artifacts(export_dir, count=preview_count)
    stats.preview_images_created = int(preview_summary["created"])
    print("EXPORT_STATS_GEMINI_BBOX:", json.dumps(asdict(stats), ensure_ascii=False))
    return stats


def parse_prepare_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Gemini-ready FineTree bbox dataset with normalized integer "
            "[ymin, xmin, ymax, xmax] coordinates."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument(
        "--instruction-mode",
        default="source",
        choices=["source", "minimal"],
        help="Use the detailed Gemini bbox prompt or a fixed minimal instruction.",
    )
    parser.add_argument(
        "--source-bbox-format",
        default=DEFAULT_SOURCE_BBOX_FORMAT,
        choices=["pixel_xywh", "normalized_1000_xywh", "normalized_1000_yxyx", "auto"],
        help="Interpretation of source annotation bbox values before Gemini normalization.",
    )
    parser.add_argument(
        "--include-doc-ids",
        default=None,
        help="Optional comma-separated document ids to include.",
    )
    parser.add_argument(
        "--validation-doc-ids",
        default=None,
        help="Optional comma-separated document ids for the validation split.",
    )
    parser.add_argument("--approved-pages-only", action="store_true", help="Export only approved pages.")
    parser.add_argument("--allow-format-issues", action="store_true", help="Do not fail on fact-format audit issues.")
    parser.add_argument("--allow-ordering-issues", action="store_true", help="Do not fail on fact-order audit issues.")
    parser.add_argument("--allow-duplicate-facts", action="store_true", help="Do not fail on duplicate-fact audit issues.")
    parser.add_argument(
        "--remove-duplicates",
        "--remove_duplicates",
        action="store_true",
        dest="remove_duplicates",
        help="Remove exact duplicate facts during Gemini export instead of failing on source duplicates.",
    )
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional minimum pixel budget for exported images.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional maximum pixel budget for exported images.")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="Number of bbox preview images to render under the export directory for sanity checking. Use 0 to disable.",
    )
    return parser.parse_args(argv)


def parse_push_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, validate, and push a Gemini-ready FineTree bbox dataset to Hugging Face "
            "with normalized integer [ymin, xmin, ymax, xmax] coordinates."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. user/fine-tree-gemini")
    parser.add_argument("--token", default=None, help="HF token (or use FINETREE_HF_TOKEN/HF_TOKEN/Doppler).")
    parser.set_defaults(public=True)
    parser.add_argument("--public", dest="public", action="store_true", help="Push the dataset as public (default).")
    parser.add_argument("--private", dest="public", action="store_false", help="Push the dataset as private.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse an existing export directory and push it as-is.")
    parser.add_argument(
        "--instruction-mode",
        default="source",
        choices=["source", "minimal"],
        help="Use the detailed Gemini bbox prompt or a fixed minimal instruction.",
    )
    parser.add_argument(
        "--source-bbox-format",
        default=DEFAULT_SOURCE_BBOX_FORMAT,
        choices=["pixel_xywh", "normalized_1000_xywh", "normalized_1000_yxyx", "auto"],
        help="Interpretation of source annotation bbox values before Gemini normalization.",
    )
    parser.add_argument(
        "--include-doc-ids",
        default=None,
        help="Optional comma-separated document ids to include.",
    )
    parser.add_argument(
        "--validation-doc-ids",
        default=None,
        help="Optional comma-separated document ids for the validation split.",
    )
    parser.add_argument("--approved-pages-only", action="store_true", help="Export only approved pages.")
    parser.add_argument("--allow-format-issues", action="store_true", help="Do not fail on fact-format audit issues.")
    parser.add_argument("--allow-ordering-issues", action="store_true", help="Do not fail on fact-order audit issues.")
    parser.add_argument("--allow-duplicate-facts", action="store_true", help="Do not fail on duplicate-fact audit issues.")
    parser.add_argument(
        "--remove-duplicates",
        "--remove_duplicates",
        action="store_true",
        dest="remove_duplicates",
        help="Remove exact duplicate facts during Gemini export instead of failing on source duplicates.",
    )
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional minimum pixel budget for exported images.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional maximum pixel budget for exported images.")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="Number of bbox preview images to render under the export directory for sanity checking. Use 0 to disable.",
    )
    return parser.parse_args(argv)


def main_prepare(argv: list[str] | None = None) -> int:
    args = parse_prepare_args(argv)
    root = Path(".").resolve()
    prepare_gemini_bbox_dataset(
        root=root,
        config_path=(root / args.config).resolve(),
        export_dir=(root / args.export_dir).resolve(),
        instruction_mode=_normalize_instruction_mode(args.instruction_mode),
        source_bbox_format=_normalize_source_bbox_format(args.source_bbox_format),
        include_doc_ids=_parse_doc_ids_csv(args.include_doc_ids) or None,
        validation_doc_ids=_parse_doc_ids_csv(args.validation_doc_ids) or None,
        approved_pages_only=bool(args.approved_pages_only),
        allow_format_issues=bool(args.allow_format_issues),
        allow_ordering_issues=bool(args.allow_ordering_issues),
        allow_duplicate_facts=bool(args.allow_duplicate_facts),
        remove_duplicates=bool(args.remove_duplicates),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        preview_count=args.preview_count,
    )
    return 0


def main_push(argv: list[str] | None = None) -> int:
    args = parse_push_args(argv)
    root = Path(".").resolve()
    export_dir = (root / args.export_dir).resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Export FINETREE_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, "
            "or HUGGINGFACEHUB_API_TOKEN."
        )

    if not args.skip_prepare:
        prepare_gemini_bbox_dataset(
            root=root,
            config_path=(root / args.config).resolve(),
            export_dir=export_dir,
            instruction_mode=_normalize_instruction_mode(args.instruction_mode),
            source_bbox_format=_normalize_source_bbox_format(args.source_bbox_format),
            include_doc_ids=_parse_doc_ids_csv(args.include_doc_ids) or None,
            validation_doc_ids=_parse_doc_ids_csv(args.validation_doc_ids) or None,
            approved_pages_only=bool(args.approved_pages_only),
            allow_format_issues=bool(args.allow_format_issues),
            allow_ordering_issues=bool(args.allow_ordering_issues),
            allow_duplicate_facts=bool(args.allow_duplicate_facts),
            remove_duplicates=bool(args.remove_duplicates),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            preview_count=args.preview_count,
        )
    else:
        validate_gemini_export_dir(export_dir)
        create_gemini_bbox_preview_artifacts(export_dir, count=args.preview_count)

    dataset, train_rows, val_rows = build_hf_dataset_from_gemini_export(export_dir)
    pushed_repo = push_gemini_bbox_dataset_to_hf(
        dataset,
        token=token,
        repo_id=str(args.repo_id).strip(),
        private=not bool(args.public),
    )
    print(
        "PUSH_STATS_GEMINI_BBOX:",
        json.dumps(
            {
                "export_dir": str(export_dir),
                "train_rows": train_rows,
                "validation_rows": val_rows,
                "pushed_repo": pushed_repo,
            },
            ensure_ascii=False,
        ),
    )
    return 0


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_SOURCE_BBOX_FORMAT",
    "GeminiBBoxExportStats",
    "MINIMAL_GEMINI_BBOX_INSTRUCTION",
    "build_gemini_bbox_prompt_template",
    "build_hf_dataset_from_gemini_export",
    "convert_bbox_to_gemini_1000",
    "create_gemini_bbox_preview_artifacts",
    "main_prepare",
    "main_push",
    "prepare_gemini_bbox_dataset",
    "push_gemini_bbox_dataset_to_hf",
    "validate_gemini_bbox_1000",
    "validate_gemini_export_dir",
    "validate_gemini_payload",
]
