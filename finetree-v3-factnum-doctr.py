#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.finetune.push_dataset_hub import resolve_hf_token  # noqa: E402


DEFAULT_SOURCE_DATASET = "asafd60/fintetree-v3-factnum"
DEFAULT_TARGET_DATASET = "asafd60/fintetree-v3-factnum-doctr"
DEFAULT_EXPORT_DIR = "artifacts/fintetree_v3_factnum_doctr"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FineTree factnum dataset into a docTR-ready folder dataset and optionally push it to HF."
    )
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--repo-id", default=DEFAULT_TARGET_DATASET)
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--token", default=None, help="HF token override.")
    parser.add_argument("--private", action="store_true", help="Push dataset privately instead of publicly.")
    parser.add_argument("--dry-run", action="store_true", help="Build the docTR-ready export locally without pushing.")
    return parser.parse_args(argv)


def sha256_file_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def xywh_pixel_to_polygon_abs(bbox: Any, width: int, height: int) -> list[list[int]] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
    except Exception:
        return None

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(x1, max(width - 1, 0)))
    x2 = max(0, min(x2, max(width - 1, 0)))
    y1 = max(0, min(y1, max(height - 1, 0)))
    y2 = max(0, min(y2, max(height - 1, 0)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _dedupe_polygons(polygons: list[list[list[int]]]) -> list[list[list[int]]]:
    deduped: list[list[list[int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for polygon in polygons:
        key = tuple((int(point[0]), int(point[1])) for point in polygon)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(polygon)
    return deduped


def build_doctr_split(hf_split: Iterable[dict[str, Any]], out_dir: Path) -> dict[str, int]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, dict[str, Any]] = {}
    processed_rows = 0
    kept_images = 0
    skipped_empty = 0
    total_polygons = 0

    for idx, example in enumerate(hf_split):
        processed_rows += 1
        image = example.get("image")
        if not isinstance(image, PILImage.Image):
            raise RuntimeError(f"Expected decoded PIL image in dataset row {idx}, got {type(image)!r}.")

        text_payload = str(example.get("text") or "")
        page_json = json.loads(text_payload)

        width, height = image.size
        polygons: list[list[list[int]]] = []
        for fact in page_json.get("facts", []):
            if not isinstance(fact, dict):
                continue
            polygon = xywh_pixel_to_polygon_abs(fact.get("bbox"), width, height)
            if polygon is not None:
                polygons.append(polygon)

        deduped = _dedupe_polygons(polygons)
        if not deduped:
            skipped_empty += 1
            continue

        filename = f"{idx:05d}.png"
        img_path = images_dir / filename
        image.save(img_path)
        img_bytes = img_path.read_bytes()

        labels[filename] = {
            "img_dimensions": [height, width],
            "img_hash": sha256_file_bytes(img_bytes),
            "polygons": deduped,
        }
        kept_images += 1
        total_polygons += len(deduped)

    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    return {
        "processed_rows": processed_rows,
        "kept_images": kept_images,
        "skipped_empty": skipped_empty,
        "polygons": total_polygons,
    }


def export_doctr_dataset(*, source_dataset: str, export_dir: Path) -> dict[str, dict[str, int]]:
    train_ds = load_dataset(source_dataset, split="train")
    validation_ds = load_dataset(source_dataset, split="validation")

    if export_dir.exists():
        for path in sorted(export_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    export_dir.mkdir(parents=True, exist_ok=True)

    train_dir = export_dir / "train"
    val_dir = export_dir / "val"

    train_stats = build_doctr_split(train_ds, train_dir)
    val_stats = build_doctr_split(validation_ds, val_dir)

    summary = {
        "source_dataset": source_dataset,
        "train": train_stats,
        "validation": val_stats,
    }
    (export_dir / "README.md").write_text(
        "# FineTree docTR Detection Export\n\n"
        f"- Source dataset: `{source_dataset}`\n"
        f"- Train kept images: {train_stats['kept_images']}\n"
        f"- Validation kept images: {val_stats['kept_images']}\n"
        f"- Train polygons: {train_stats['polygons']}\n"
        f"- Validation polygons: {val_stats['polygons']}\n"
        f"- Skipped empty train rows: {train_stats['skipped_empty']}\n"
        f"- Skipped empty validation rows: {val_stats['skipped_empty']}\n",
        encoding="utf-8",
    )
    (export_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def push_export(*, export_dir: Path, repo_id: str, token: str, private: bool) -> str:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(export_dir),
        commit_message="Upload docTR-ready FineTree dataset",
    )
    return repo_id


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_dir = (Path.cwd().resolve() / args.export_dir).resolve()
    summary = export_doctr_dataset(source_dataset=str(args.source_dataset), export_dir=export_dir)

    print(f"SOURCE_DATASET: {args.source_dataset}")
    print(f"EXPORT_DIR: {export_dir}")
    print(f"TRAIN_ROWS: {summary['train']['processed_rows']}")
    print(f"TRAIN_IMAGES: {summary['train']['kept_images']}")
    print(f"TRAIN_SKIPPED_EMPTY: {summary['train']['skipped_empty']}")
    print(f"TRAIN_POLYGONS: {summary['train']['polygons']}")
    print(f"VAL_ROWS: {summary['validation']['processed_rows']}")
    print(f"VAL_IMAGES: {summary['validation']['kept_images']}")
    print(f"VAL_SKIPPED_EMPTY: {summary['validation']['skipped_empty']}")
    print(f"VAL_POLYGONS: {summary['validation']['polygons']}")
    print(f"REPO_ID: {args.repo_id}")
    print(f"PUBLIC: {not args.private}")

    if args.dry_run:
        print("DRY_RUN: true")
        return 0

    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError("Missing HF token. Pass --token or export FINETREE_HF_TOKEN/HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")

    pushed_repo = push_export(
        export_dir=export_dir,
        repo_id=str(args.repo_id),
        token=token,
        private=bool(args.private),
    )
    print(f"PUSHED_REPO: {pushed_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
