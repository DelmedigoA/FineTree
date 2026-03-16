#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageDraw

from finetree_annotator.finetune.push_dataset_hub import (
    _copy_or_resize_image,
    _iter_fact_dicts,
    _resolve_resize_bounds,
    _resolve_system_prompt,
    _rows_from_chat_jsonl,
    _scale_bbox_text_payload,
)


def _resolve_row_image(root: Path, row: dict[str, str]) -> Path:
    image_path = Path(row["image"])
    if image_path.is_absolute():
        return image_path
    return (root / image_path).resolve()


def _payload_bbox_count(text: str) -> int:
    try:
        payload = json.loads(text)
    except Exception:
        return 0
    count = 0
    if isinstance(payload, dict):
        for fact in _iter_fact_dicts(payload):
            bbox = fact.get("bbox")
            if isinstance(bbox, dict):
                count += 1
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                count += 1
    return count


def _select_rows(root: Path, split: str, count: int) -> list[dict[str, str]]:
    split_path = root / "data" / "finetune" / f"{split}.jsonl"
    system_prompt = _resolve_system_prompt(root)
    selected: list[dict[str, str]] = []
    for row in _rows_from_chat_jsonl(split_path, system_prompt=system_prompt):
        if _payload_bbox_count(row["text"]) <= 0:
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _draw_bboxes(image_path: Path, text: str, output_path: Path) -> int:
    payload = json.loads(text)
    with PILImage.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    drawn = 0
    for fact in _iter_fact_dicts(payload):
        bbox = fact.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        x0 = int(x)
        y0 = int(y)
        x1 = int(x + w)
        y1 = int(y + h)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=2)
        drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return drawn


def inspect_rows(
    root: Path,
    *,
    split: str,
    count: int,
    min_pixels: int | None,
    max_pixels: int | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    selected_rows = _select_rows(root, split, count)
    if len(selected_rows) < count:
        raise RuntimeError(f"Requested {count} rows but found only {len(selected_rows)} rows with bbox annotations in {split}.")

    resize_bounds = _resolve_resize_bounds(min_pixels, max_pixels)
    resolved_min = resize_bounds[0] if resize_bounds else None
    resolved_max = resize_bounds[1] if resize_bounds else None

    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        src_img = _resolve_row_image(root, row)
        doc_id = src_img.parent.name or "unknown_doc"
        page_name = src_img.stem
        sample_dir = output_dir / f"{index:02d}_{doc_id}_{page_name}"
        resized_dir = sample_dir / "resized"
        resized_dir.mkdir(parents=True, exist_ok=True)

        resized_image_path = resized_dir / src_img.name
        resize_stats = _copy_or_resize_image(
            src_img,
            resized_image_path,
            min_pixels=resolved_min,
            max_pixels=resolved_max,
        )
        scale_x = float(resize_stats["new_w"]) / max(float(resize_stats["orig_w"]), 1.0)
        scale_y = float(resize_stats["new_h"]) / max(float(resize_stats["orig_h"]), 1.0)
        scaled_text, updated, clamped, parse_fail = _scale_bbox_text_payload(
            row["text"],
            scale_x=scale_x,
            scale_y=scale_y,
            new_w=int(resize_stats["new_w"]),
            new_h=int(resize_stats["new_h"]),
        )
        if parse_fail:
            raise RuntimeError(f"Failed to parse assistant payload for {src_img}.")

        overlay_path = sample_dir / f"{page_name}__bbox_overlay.png"
        drawn = _draw_bboxes(resized_image_path, scaled_text, overlay_path)
        payload = json.loads(scaled_text)
        meta = payload.get("meta") if isinstance(payload, dict) else {}
        facts = payload.get("facts") if isinstance(payload, dict) else []
        result = {
            "index": index,
            "document_id": doc_id,
            "page": src_img.name,
            "source_image": str(src_img),
            "resized_image": str(resized_image_path),
            "overlay_image": str(overlay_path),
            "orig_size": [int(resize_stats["orig_w"]), int(resize_stats["orig_h"])],
            "resized_size": [int(resize_stats["new_w"]), int(resize_stats["new_h"])],
            "orig_pixels": int(resize_stats["orig_w"]) * int(resize_stats["orig_h"]),
            "resized_pixels": int(resize_stats["new_w"]) * int(resize_stats["new_h"]),
            "scale_x": scale_x,
            "scale_y": scale_y,
            "bbox_updated": updated,
            "bbox_clamped": clamped,
            "bbox_drawn": drawn,
            "page_meta": meta,
            "fact_count": len(facts) if isinstance(facts, list) else 0,
        }
        (sample_dir / "scaled_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_dir / "inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HF export bbox scaling by drawing transformed boxes on resized images.")
    parser.add_argument("--root", type=Path, default=Path(".").resolve(), help="Repository root.")
    parser.add_argument("--split", choices=("train", "val"), default="train", help="Dataset split to inspect.")
    parser.add_argument("--count", type=int, default=3, help="Number of samples to inspect.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional minimum pixel budget.")
    parser.add_argument("--max-pixels", type=int, default=1_000_000, help="Maximum pixel budget to match HF export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hf_bbox_inspection/max_pixels_1000000"),
        help="Directory for inspection artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    results = inspect_rows(
        root,
        split=args.split,
        count=args.count,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        output_dir=output_dir,
    )
    summary = {
        "split": args.split,
        "count": len(results),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "output_dir": str(output_dir.resolve()),
        "samples": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
