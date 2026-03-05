from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi
from PIL import Image as PILImage

from ..fact_normalization import assert_fact_format
from ..fact_ordering import assert_fact_order
from .config import load_finetune_config
from .duplicate_facts import assert_no_duplicate_facts

_MIN_BBOX_SIZE = 1.0
InstructionMode = Literal["source", "minimal"]
_MINIMAL_INSTRUCTION = "Extract metadata and financial facts from the provided image."
_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise financial statement extraction system. "
    "Return only valid JSON that matches the required schema."
)
_REQUIRED_CANONICAL_PROMPT_KEYS: tuple[str, ...] = ("comment", "is_note", "note", "note_reference")
_LEGACY_PROMPT_KEYS: tuple[str, ...] = ("is_beur", "beur_num", "refference")
_COMPACT_KEY_MAP: dict[str, str] = {
    "meta": "m",
    "facts": "f",
    "bbox": "b",
    "entity_name": "e",
    "page_num": "p",
    "title": "ttl",
    "value": "v",
    "comment": "cmt",
    "is_note": "isn",
    "note": "nt",
    "note_reference": "nref",
    "date": "d",
    "currency": "cur",
    "scale": "sc",
    "value_type": "vt",
    # legacy aliases kept for backwards compatibility in compaction paths
    "is_beur": "isn",
    "beur_num": "nt",
    "reference": "ref",
    "refference": "ref",
}


def build_dataset(config_path: Path, *, allow_format_issues: bool = False) -> None:
    from .dataset_builder import main as build_main

    args = ["--config", str(config_path)]
    if allow_format_issues:
        args.append("--allow-format-issues")
    build_main(args)


def _resolve_system_prompt(root: Path) -> str:
    prompt_path = (root / "system_prompt.txt").resolve()
    if prompt_path.is_file():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return _DEFAULT_SYSTEM_PROMPT


def _rows_from_chat_jsonl(
    split_path: Path,
    *,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    if not split_path.is_file():
        return []

    rows: list[dict[str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
        if len(messages) < 2:
            continue

        user_content = messages[0].get("content") if isinstance(messages[0], dict) else None
        assistant_content = messages[1].get("content") if isinstance(messages[1], dict) else None
        if not isinstance(user_content, list) or not isinstance(assistant_content, list):
            continue

        image_path = ""
        instruction = ""
        text = ""

        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and isinstance(part.get("image"), str):
                image_path = part["image"]
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                instruction = part["text"]

        if assistant_content and isinstance(assistant_content[0], dict):
            text_val = assistant_content[0].get("text")
            if isinstance(text_val, str):
                text = text_val

        if image_path and instruction and text:
            rows.append(
                {
                    "image": image_path,
                    "system": system_prompt,
                    "instruction": instruction,
                    "text": text,
                }
            )
    return rows


def _has_prompt_key(text: str, key: str) -> bool:
    return re.search(rf"\b{re.escape(key)}\b", text, flags=re.IGNORECASE) is not None


def assert_source_instruction_schema(
    root: Path,
    *,
    fail_on_issues: bool = True,
    train_path: Path | None = None,
    val_path: Path | None = None,
) -> dict[str, Any]:
    resolved_train = train_path or (root / "data/finetune/train.jsonl")
    resolved_val = val_path or (root / "data/finetune/val.jsonl")
    checked_rows = 0
    rows_with_issues = 0
    issue_examples: list[dict[str, Any]] = []

    for split_name, split_path in (("train", resolved_train), ("validation", resolved_val)):
        if not split_path.is_file():
            continue
        for line_num, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue
            if not isinstance(sample, dict):
                continue
            _doc_id, _image_path, instruction, _text = _sample_from_chat_messages(sample, root=root)
            if not instruction:
                continue
            checked_rows += 1
            issues: list[str] = []
            for key in _REQUIRED_CANONICAL_PROMPT_KEYS:
                if not _has_prompt_key(instruction, key):
                    issues.append(f"missing_required_key:{key}")
            for legacy in _LEGACY_PROMPT_KEYS:
                if _has_prompt_key(instruction, legacy):
                    issues.append(f"legacy_key_present:{legacy}")

            if issues:
                rows_with_issues += 1
                if len(issue_examples) < 20:
                    issue_examples.append(
                        {
                            "split": split_name,
                            "line": line_num,
                            "issues": issues,
                        }
                    )

    report = {
        "checked_rows": checked_rows,
        "rows_with_issues": rows_with_issues,
        "required_keys": list(_REQUIRED_CANONICAL_PROMPT_KEYS),
        "legacy_keys": list(_LEGACY_PROMPT_KEYS),
        "issue_examples": issue_examples,
    }
    print("PROMPT_SCHEMA_AUDIT:", json.dumps(report, ensure_ascii=False))
    if fail_on_issues and rows_with_issues > 0:
        raise RuntimeError(
            "Source prompt schema validation failed for one or more finetune rows. "
            "Ensure prompt instructions use canonical keys "
            "comment/is_note/note/note_reference and remove legacy keys."
        )
    return report


def _sample_from_chat_messages(sample: dict[str, Any], *, root: Path) -> tuple[str | None, str | None, str | None, str | None]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    doc_id = metadata.get("document_id") if isinstance(metadata.get("document_id"), str) else None

    image_path: str | None = None
    instruction: str | None = None
    text: str | None = None

    messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
    if len(messages) >= 2 and isinstance(messages[0], dict) and isinstance(messages[1], dict):
        user_content = messages[0].get("content") if isinstance(messages[0].get("content"), list) else []
        assistant_content = messages[1].get("content") if isinstance(messages[1].get("content"), list) else []

        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and isinstance(part.get("image"), str):
                raw = str(part["image"]).strip()
                if raw:
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = (root / candidate).resolve()
                    image_path = str(candidate)
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                val = str(part["text"]).strip()
                if val:
                    instruction = val

        if assistant_content and isinstance(assistant_content[0], dict):
            raw_text = assistant_content[0].get("text")
            if isinstance(raw_text, str):
                val = raw_text.strip()
                if val:
                    text = val

    return doc_id, image_path, instruction, text


def _split_signatures(split_path: Path, *, root: Path) -> dict[str, set[str] | int]:
    doc_ids: set[str] = set()
    image_paths: set[str] = set()
    sample_hashes: set[str] = set()
    rows = 0

    if not split_path.is_file():
        return {
            "doc_ids": doc_ids,
            "image_paths": image_paths,
            "sample_hashes": sample_hashes,
            "rows": rows,
        }

    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            sample = json.loads(line)
        except Exception:
            continue
        if not isinstance(sample, dict):
            continue
        doc_id, image_path, instruction, text = _sample_from_chat_messages(sample, root=root)
        if doc_id:
            doc_ids.add(doc_id)
        if image_path:
            image_paths.add(image_path)
        if any(v is not None for v in (image_path, instruction, text)):
            payload = {
                "image": image_path or "",
                "instruction": instruction or "",
                "text": text or "",
            }
            payload_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            sample_hashes.add(hashlib.sha256(payload_str.encode("utf-8")).hexdigest())

    return {
        "doc_ids": doc_ids,
        "image_paths": image_paths,
        "sample_hashes": sample_hashes,
        "rows": rows,
    }


def assert_no_train_val_contamination(
    root: Path,
    *,
    train_path: Path | None = None,
    val_path: Path | None = None,
) -> dict[str, Any]:
    resolved_train = train_path or (root / "data/finetune/train.jsonl")
    resolved_val = val_path or (root / "data/finetune/val.jsonl")

    train = _split_signatures(resolved_train, root=root)
    val = _split_signatures(resolved_val, root=root)

    train_doc_ids = train["doc_ids"] if isinstance(train["doc_ids"], set) else set()
    val_doc_ids = val["doc_ids"] if isinstance(val["doc_ids"], set) else set()
    train_images = train["image_paths"] if isinstance(train["image_paths"], set) else set()
    val_images = val["image_paths"] if isinstance(val["image_paths"], set) else set()
    train_hashes = train["sample_hashes"] if isinstance(train["sample_hashes"], set) else set()
    val_hashes = val["sample_hashes"] if isinstance(val["sample_hashes"], set) else set()

    overlap_doc_ids = sorted(train_doc_ids & val_doc_ids)
    overlap_images = sorted(train_images & val_images)
    overlap_samples = sorted(train_hashes & val_hashes)

    report = {
        "train_rows": int(train["rows"]) if isinstance(train["rows"], int) else 0,
        "val_rows": int(val["rows"]) if isinstance(val["rows"], int) else 0,
        "train_unique_doc_ids": len(train_doc_ids),
        "val_unique_doc_ids": len(val_doc_ids),
        "train_unique_images": len(train_images),
        "val_unique_images": len(val_images),
        "overlap_doc_ids": len(overlap_doc_ids),
        "overlap_images": len(overlap_images),
        "overlap_samples": len(overlap_samples),
        "overlap_doc_ids_examples": overlap_doc_ids[:10],
        "overlap_image_examples": overlap_images[:10],
    }
    print("SPLIT_AUDIT:", json.dumps(report, ensure_ascii=False))

    if overlap_doc_ids or overlap_images or overlap_samples:
        raise RuntimeError(
            "Train/validation contamination detected. "
            f"overlap_doc_ids={len(overlap_doc_ids)} "
            f"overlap_images={len(overlap_images)} "
            f"overlap_samples={len(overlap_samples)}"
        )
    return report


def _resolve_resize_bounds(min_pixels: int | None, max_pixels: int | None) -> tuple[int | None, int | None] | None:
    if min_pixels is None and max_pixels is None:
        return None
    if min_pixels is not None and int(min_pixels) <= 0:
        raise ValueError("--min-pixels must be > 0.")
    if max_pixels is not None and int(max_pixels) <= 0:
        raise ValueError("--max-pixels must be > 0.")
    if min_pixels is not None and max_pixels is not None and int(min_pixels) > int(max_pixels):
        raise ValueError("--min-pixels cannot be greater than --max-pixels.")
    return int(min_pixels) if min_pixels is not None else None, int(max_pixels) if max_pixels is not None else None


def _normalize_instruction_mode(value: str) -> InstructionMode:
    mode = (value or "").strip().lower()
    if mode not in {"source", "minimal"}:
        raise ValueError("--instruction-mode must be one of: source, minimal")
    return mode  # type: ignore[return-value]


def _parse_excluded_doc_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _compact_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _remove_numeric_thousands_separators(value: str) -> str:
    stripped = value.strip()
    if "," not in stripped:
        return value

    accounting_negative = stripped.startswith("(") and stripped.endswith(")")
    if ("(" in stripped or ")" in stripped) and not accounting_negative:
        return value

    core = stripped[1:-1] if accounting_negative else stripped
    if not core:
        return value

    sign = ""
    if core[0] in {"+", "-"}:
        sign = core[0]
        core = core[1:]
    if not core:
        return value

    if core.count(".") > 1:
        return value
    int_part, dot, frac_part = core.partition(".")
    if not int_part:
        return value

    groups = int_part.split(",")
    if len(groups) <= 1:
        return value
    if not groups[0].isdigit() or len(groups[0]) > 3:
        return value
    if any((not g.isdigit() or len(g) != 3) for g in groups[1:]):
        return value

    normalized = sign + "".join(groups) + (dot + frac_part if dot else "")
    return f"({normalized})" if accounting_negative else normalized


def _compact_payload_tokens(payload: Any, *, remap_keys: bool = False) -> Any:
    if isinstance(payload, dict):
        compacted: dict[str, Any] = {}
        for key, value in payload.items():
            mapped_key = _COMPACT_KEY_MAP.get(str(key), str(key)) if remap_keys else str(key)
            compacted_value = _compact_payload_tokens(value, remap_keys=remap_keys)
            if mapped_key in compacted and mapped_key == "ref":
                existing = compacted[mapped_key]
                if (existing is None or existing == "") and compacted_value not in {None, ""}:
                    compacted[mapped_key] = compacted_value
                continue
            compacted[mapped_key] = compacted_value
        return compacted
    if isinstance(payload, list):
        return [_compact_payload_tokens(item, remap_keys=remap_keys) for item in payload]
    if isinstance(payload, str):
        return _remove_numeric_thousands_separators(payload)
    return payload


def _compact_token_text_payload(text: str, *, remap_keys: bool = False) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return text
    compacted_payload = _compact_payload_tokens(payload, remap_keys=remap_keys)
    return _compact_json_dumps(compacted_payload)


def _smart_resize_dimensions(height: int, width: int, *, min_pixels: int | None, max_pixels: int | None) -> tuple[int, int]:
    try:
        from qwen_vl_utils.vision_process import smart_resize
    except Exception as exc:
        raise RuntimeError(
            "Image resizing requested but qwen_vl_utils is not available. Install with `pip install qwen-vl-utils`."
        ) from exc

    kwargs: dict[str, int] = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
    sig = inspect.signature(smart_resize)
    factor_param = sig.parameters.get("factor")
    if factor_param is not None and factor_param.default is inspect._empty:
        # qwen_vl_utils versions with explicit factor require the Qwen vision grid multiple.
        kwargs["factor"] = 28
    new_h, new_w = smart_resize(int(height), int(width), **kwargs)
    return int(new_h), int(new_w)


def _clamp_bbox(x: float, y: float, w: float, h: float, *, image_w: int, image_h: int) -> tuple[float, float, float, float, bool]:
    clamped = False

    x0 = max(0.0, float(x))
    y0 = max(0.0, float(y))
    if x0 != x or y0 != y:
        clamped = True

    max_x = max(float(image_w) - _MIN_BBOX_SIZE, 0.0)
    max_y = max(float(image_h) - _MIN_BBOX_SIZE, 0.0)
    x1 = min(x0, max_x)
    y1 = min(y0, max_y)
    if x1 != x0 or y1 != y0:
        clamped = True

    max_w = max(float(image_w) - x1, _MIN_BBOX_SIZE)
    max_h = max(float(image_h) - y1, _MIN_BBOX_SIZE)
    w0 = max(float(w), _MIN_BBOX_SIZE)
    h0 = max(float(h), _MIN_BBOX_SIZE)
    if w0 != w or h0 != h:
        clamped = True
    w1 = min(w0, max_w)
    h1 = min(h0, max_h)
    if w1 != w0 or h1 != h0:
        clamped = True

    return x1, y1, w1, h1, clamped


def _scale_bbox_text_payload(
    text: str,
    *,
    scale_x: float,
    scale_y: float,
    new_w: int,
    new_h: int,
) -> tuple[str, int, int, int]:
    try:
        payload = json.loads(text)
    except Exception:
        return text, 0, 0, 1

    updated = 0
    clamped = 0
    if isinstance(payload, dict):
        facts = payload.get("facts")
        if isinstance(facts, list):
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                raw_bbox = fact.get("bbox")
                if isinstance(raw_bbox, dict):
                    x_raw = raw_bbox.get("x")
                    y_raw = raw_bbox.get("y")
                    w_raw = raw_bbox.get("w")
                    h_raw = raw_bbox.get("h")
                elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
                    x_raw, y_raw, w_raw, h_raw = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
                else:
                    continue
                try:
                    x = float(x_raw) * scale_x
                    y = float(y_raw) * scale_y
                    w = float(w_raw) * scale_x
                    h = float(h_raw) * scale_y
                except Exception:
                    continue
                x, y, w, h, is_clamped = _clamp_bbox(x, y, w, h, image_w=new_w, image_h=new_h)
                # Token-lean quantization for exported bbox:
                # position rounds down; size rounds up to avoid under-covering the target region.
                x_i = max(0, min(int(math.floor(x)), max(new_w - 1, 0)))
                y_i = max(0, min(int(math.floor(y)), max(new_h - 1, 0)))
                w_i = max(int(math.ceil(w)), int(_MIN_BBOX_SIZE))
                h_i = max(int(math.ceil(h)), int(_MIN_BBOX_SIZE))
                max_w = max(new_w - x_i, int(_MIN_BBOX_SIZE))
                max_h = max(new_h - y_i, int(_MIN_BBOX_SIZE))
                if w_i > max_w:
                    w_i = max_w
                    is_clamped = True
                if h_i > max_h:
                    h_i = max_h
                    is_clamped = True

                fact["bbox"] = [x_i, y_i, w_i, h_i]
                updated += 1
                if is_clamped:
                    clamped += 1

    return json.dumps(payload, ensure_ascii=False), updated, clamped, 0


def _copy_or_resize_image(
    src_img: Path,
    dst_img: Path,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    resize_enabled = min_pixels is not None or max_pixels is not None
    with PILImage.open(src_img) as img:
        orig_w, orig_h = img.size
        new_w = orig_w
        new_h = orig_h
        if resize_enabled:
            new_h, new_w = _smart_resize_dimensions(orig_h, orig_w, min_pixels=min_pixels, max_pixels=max_pixels)
            new_h = max(1, int(new_h))
            new_w = max(1, int(new_w))
        resized = (new_w != orig_w) or (new_h != orig_h)
        if resized:
            out_img = img.resize((new_w, new_h), PILImage.Resampling.BICUBIC)
            out_img.save(dst_img)
        else:
            shutil.copy2(src_img, dst_img)

    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "new_w": new_w,
        "new_h": new_h,
        "resized": resized,
    }


def _rows_from_export_jsonl(
    split_path: Path,
    export_dir: Path,
    *,
    default_system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    if not split_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if not isinstance(sample, dict):
            continue
        image_path = sample.get("image")
        system_prompt = sample.get("system")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            system_prompt = sample.get("system_prompt")
        instruction = sample.get("instruction")
        text = sample.get("text")
        if not isinstance(image_path, str) or not isinstance(instruction, str) or not isinstance(text, str):
            continue
        resolved_system_prompt = (
            str(system_prompt).strip()
            if isinstance(system_prompt, str) and str(system_prompt).strip()
            else default_system_prompt
        )
        abs_img = (export_dir / image_path).resolve()
        rows.append(
            {
                "image": str(abs_img),
                "system": resolved_system_prompt,
                "instruction": instruction,
                "text": text,
            }
        )
    return rows


def _build_hf_dataset_from_rows(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> tuple[DatasetDict, int, int]:
    features = Features(
        {
            "image": Image(),
            "system": Value("string"),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    empty_payload = {"image": [], "system": [], "instruction": [], "text": []}
    train_ds = Dataset.from_list(train_rows, features=features) if train_rows else Dataset.from_dict(empty_payload, features=features)
    val_ds = Dataset.from_list(val_rows, features=features) if val_rows else Dataset.from_dict(empty_payload, features=features)
    return DatasetDict({"train": train_ds, "validation": val_ds}), len(train_rows), len(val_rows)


def build_hf_dataset(root: Path) -> tuple[DatasetDict, int, int]:
    system_prompt = _resolve_system_prompt(root)
    train_rows = _rows_from_chat_jsonl(root / "data/finetune/train.jsonl", system_prompt=system_prompt)
    val_rows = _rows_from_chat_jsonl(root / "data/finetune/val.jsonl", system_prompt=system_prompt)
    return _build_hf_dataset_from_rows(train_rows, val_rows)


def build_hf_dataset_from_export(export_dir: Path) -> tuple[DatasetDict, int, int]:
    default_system_prompt = _resolve_system_prompt(Path(".").resolve())
    train_rows = _rows_from_export_jsonl(
        export_dir / "train.jsonl",
        export_dir,
        default_system_prompt=default_system_prompt,
    )
    val_rows = _rows_from_export_jsonl(
        export_dir / "val.jsonl",
        export_dir,
        default_system_prompt=default_system_prompt,
    )
    return _build_hf_dataset_from_rows(train_rows, val_rows)


def export_for_hf(
    root: Path,
    export_dir: Path,
    *,
    instruction_mode: InstructionMode = "source",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    exclude_doc_ids: set[str] | None = None,
    compact_tokens: bool = False,
    aggressive_compact_tokens: bool = False,
) -> Tuple[int, int]:
    instruction_mode = _normalize_instruction_mode(instruction_mode)
    system_prompt = _resolve_system_prompt(root)
    resize_bounds = _resolve_resize_bounds(min_pixels, max_pixels)
    resolved_min = resize_bounds[0] if resize_bounds else None
    resolved_max = resize_bounds[1] if resize_bounds else None
    resize_enabled = resize_bounds is not None
    excluded = set(exclude_doc_ids or set())

    train_in = root / "data/finetune/train.jsonl"
    val_in = root / "data/finetune/val.jsonl"

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    resized_images = 0
    unchanged_images = 0
    bbox_updated = 0
    bbox_clamped = 0
    json_parse_failures = 0
    skipped_excluded_docs = 0

    def export_split(src_path: Path, dst_name: str) -> int:
        nonlocal resized_images, unchanged_images, bbox_updated, bbox_clamped, json_parse_failures, skipped_excluded_docs
        dst_path = export_dir / dst_name
        if not src_path.exists():
            dst_path.write_text("", encoding="utf-8")
            return 0

        out_lines: list[str] = []
        for row in _rows_from_chat_jsonl(src_path, system_prompt=system_prompt):
            src_img = Path(row["image"])
            if not src_img.is_absolute():
                src_img = (root / src_img).resolve()
            if not src_img.exists():
                raise FileNotFoundError(f"Missing source image: {src_img}")

            doc_id = src_img.parent.name or "unknown_doc"
            if doc_id in excluded:
                skipped_excluded_docs += 1
                continue
            rel_img = Path("images") / doc_id / src_img.name
            dst_img = export_dir / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            resize_stats = _copy_or_resize_image(
                src_img,
                dst_img,
                min_pixels=resolved_min,
                max_pixels=resolved_max,
            )
            if resize_stats["resized"]:
                resized_images += 1
            else:
                unchanged_images += 1

            text_out = row["text"]
            scale_x = float(resize_stats["new_w"]) / max(float(resize_stats["orig_w"]), 1.0)
            scale_y = float(resize_stats["new_h"]) / max(float(resize_stats["orig_h"]), 1.0)
            text_out, updated, clamped, parse_fail = _scale_bbox_text_payload(
                text_out,
                scale_x=scale_x,
                scale_y=scale_y,
                new_w=int(resize_stats["new_w"]),
                new_h=int(resize_stats["new_h"]),
            )
            if compact_tokens or aggressive_compact_tokens:
                text_out = _compact_token_text_payload(
                    text_out,
                    remap_keys=aggressive_compact_tokens,
                )
            bbox_updated += updated
            bbox_clamped += clamped
            json_parse_failures += parse_fail

            out_lines.append(
                json.dumps(
                    {
                        "image": rel_img.as_posix(),
                        "system": row["system"],
                        "instruction": (
                            _MINIMAL_INSTRUCTION
                            if instruction_mode == "minimal"
                            else row["instruction"].replace(str(src_img), rel_img.as_posix())
                        ),
                        "text": text_out,
                    },
                    ensure_ascii=False,
                )
            )

        dst_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        return len(out_lines)

    train_rows = export_split(train_in, "train.jsonl")
    val_rows = export_split(val_in, "val.jsonl")

    (export_dir / "README.md").write_text(
        "# FineTree Annotated Dataset\n\n"
        "Generated from current repository annotations.\n\n"
        "- Includes `system` column from `system_prompt.txt`.\n"
        f"- Instruction mode: {instruction_mode}\n"
        f"- Resize enabled: {'yes' if resize_enabled else 'no'}\n"
        f"- Min pixels: {resolved_min if resolved_min is not None else 'unset'}\n"
        f"- Max pixels: {resolved_max if resolved_max is not None else 'unset'}\n"
        f"- Resized images: {resized_images}\n"
        f"- Unchanged images: {unchanged_images}\n"
        f"- BBox updated: {bbox_updated}\n"
        f"- BBox clamped: {bbox_clamped}\n"
        f"- BBox JSON parse failures: {json_parse_failures}\n"
        f"- Skipped rows by excluded docs: {skipped_excluded_docs}\n"
        f"- Excluded doc ids: {sorted(excluded)}\n"
        f"- Compact tokens: {'yes' if compact_tokens else 'no'}\n"
        f"- Aggressive compact tokens: {'yes' if aggressive_compact_tokens else 'no'}\n"
        f"- Train rows: {train_rows}\n"
        f"- Val rows: {val_rows}\n",
        encoding="utf-8",
    )
    print(
        "EXPORT_STATS:",
        json.dumps(
            {
                "instruction_mode": instruction_mode,
                "resize_enabled": resize_enabled,
                "min_pixels": resolved_min,
                "max_pixels": resolved_max,
                "resized_images": resized_images,
                "unchanged_images": unchanged_images,
                "bbox_updated": bbox_updated,
                "bbox_clamped": bbox_clamped,
                "bbox_json_parse_failures": json_parse_failures,
                "skipped_excluded_docs": skipped_excluded_docs,
                "excluded_doc_ids": sorted(excluded),
                "compact_tokens": compact_tokens,
                "aggressive_compact_tokens": aggressive_compact_tokens,
            },
            ensure_ascii=False,
        ),
    )
    return train_rows, val_rows


def _default_repo_id(api: HfApi) -> str:
    username = api.whoami().get("name") or "user"
    return f"{username}/FineTree-annotated-pages"


def _resolve_doppler_scope_args() -> list[str]:
    args: list[str] = []
    project = (os.getenv("DOPPLER_PROJECT") or "").strip()
    config = (os.getenv("DOPPLER_CONFIG") or "").strip()
    if project:
        args.extend(["--project", project])
    if config:
        args.extend(["--config", config])
    return args


def _token_from_doppler() -> Optional[str]:
    if shutil.which("doppler") is None:
        return None
    scope_args = _resolve_doppler_scope_args()
    for secret in ("HF_TOKEN", "FINETREE_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        cmd = ["doppler", "secrets", "get", secret, "--plain", *scope_args]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        token = proc.stdout.strip()
        if token:
            return token
    return None


def resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    return (
        (explicit_token or "").strip()
        or (os.getenv("FINETREE_HF_TOKEN") or "").strip()
        or (os.getenv("HF_TOKEN") or "").strip()
        or (os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
        or (os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
        or _token_from_doppler()
    )


def _dataset_for_push(dataset: DatasetDict) -> tuple[DatasetDict, list[str], list[str]]:
    kept = [split for split in dataset.keys() if len(dataset[split]) > 0]
    dropped = [split for split in dataset.keys() if split not in kept]
    if not kept:
        raise RuntimeError("No non-empty dataset splits to push after filtering.")
    if not dropped:
        return dataset, kept, dropped
    filtered = DatasetDict({split: dataset[split] for split in kept})
    return filtered, kept, dropped


def push_to_hf(dataset: DatasetDict, token: str, repo_id: str | None, private: bool = True) -> str:
    api = HfApi(token=token)
    if repo_id is None:
        repo_id = _default_repo_id(api)

    dataset_to_push, kept_splits, dropped_splits = _dataset_for_push(dataset)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    # Push split-by-split to avoid DatasetDict-level upload bugs in some datasets versions.
    for split in kept_splits:
        dataset_to_push[split].push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            split=split,
        )
    print(f"PUSH_SPLITS: kept={kept_splits} dropped={dropped_splits}")
    return repo_id


def _derive_variant_repo_ids(base_repo_id: str | None) -> tuple[str | None, str | None, str | None]:
    if not base_repo_id:
        return None, None, None
    repo = str(base_repo_id).strip()
    if not repo:
        return None, None, None
    if "/" in repo:
        owner, name = repo.split("/", 1)
        owner = owner.strip()
        name = name.strip()
        if not owner or not name:
            return None, None, None
        prefix = f"{owner}/{name}"
    else:
        name = repo
        if not name:
            return None, None, None
        prefix = name
    return (
        f"{prefix}-minimal-instruction",
        f"{prefix}-no-bbox",
        f"{prefix}-no-bbox-minimal-instruction",
    )


def _resolve_repo_id(repo_id: str | None, token: str) -> str:
    if repo_id:
        return repo_id
    api = HfApi(token=token)
    return _default_repo_id(api)


def _split_repo_ids(
    base_repo_id: str,
    *,
    repo_id_train: str | None = None,
    repo_id_validation: str | None = None,
) -> tuple[str, str]:
    return (
        repo_id_train or f"{base_repo_id}-train",
        repo_id_validation or f"{base_repo_id}-validation",
    )


def push_train_validation_separately(
    dataset: DatasetDict,
    *,
    token: str,
    base_repo_id: str,
    private: bool = True,
    repo_id_train: str | None = None,
    repo_id_validation: str | None = None,
) -> dict[str, str]:
    dataset_to_push, kept_splits, dropped_splits = _dataset_for_push(dataset)
    train_repo, val_repo = _split_repo_ids(
        base_repo_id,
        repo_id_train=repo_id_train,
        repo_id_validation=repo_id_validation,
    )

    api = HfApi(token=token)
    pushed: dict[str, str] = {}

    if "train" in kept_splits:
        api.create_repo(repo_id=train_repo, repo_type="dataset", private=private, exist_ok=True)
        dataset_to_push["train"].push_to_hub(
            repo_id=train_repo,
            token=token,
            private=private,
            split="train",
        )
        pushed["train"] = train_repo

    if "validation" in kept_splits:
        api.create_repo(repo_id=val_repo, repo_type="dataset", private=private, exist_ok=True)
        # Keep split name "train" for the validation-only repo to simplify trainer configs.
        dataset_to_push["validation"].push_to_hub(
            repo_id=val_repo,
            token=token,
            private=private,
            split="train",
        )
        pushed["validation"] = val_repo

    print(f"PUSH_SPLIT_REPOS: pushed={pushed} dropped={dropped_splits}")
    return pushed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push FineTree dataset with columns: image, instruction, text.")
    parser.add_argument("--config", default="configs/finetune_qwen35a3_vl.yaml")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id, e.g. username/FineTree-annotated-pages")
    parser.add_argument(
        "--repo-id-minimal-instruction",
        default=None,
        help="HF repo id for bbox minimal-instruction variant.",
    )
    parser.add_argument("--repo-id-no-bbox", default=None, help="HF repo id for no-bbox source-instruction variant.")
    parser.add_argument(
        "--repo-id-no-bbox-minimal-instruction",
        default=None,
        help="HF repo id for no-bbox minimal-instruction variant.",
    )
    parser.add_argument("--token", default=None, help="HF token (or use FINETREE_HF_TOKEN/HF_TOKEN/Doppler)")
    parser.add_argument("--export-dir", default="artifacts/hf_dataset_export")
    parser.add_argument("--public", action="store_true", help="Create/push dataset as public.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional minimum image pixel budget for export.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional maximum image pixel budget for export.")
    parser.add_argument(
        "--instruction-mode",
        default="source",
        choices=["source", "minimal"],
        help="Use source instruction or fixed minimal instruction for the bbox dataset push.",
    )
    parser.add_argument(
        "--exclude-doc-ids",
        default=None,
        help="Comma-separated document/image folder ids to exclude, e.g. pdf_4,test",
    )
    parser.add_argument(
        "--compact_tokens",
        action="store_true",
        help="Compact assistant JSON payload formatting and numeric separators without renaming keys.",
    )
    parser.add_argument(
        "--aggressive-compact-tokens",
        "--aggressive_compact_tokens",
        action="store_true",
        dest="aggressive_compact_tokens",
        help="Aggressive compaction: enables key shortening in addition to --compact_tokens behavior.",
    )
    parser.add_argument("--push-all-variants", action="store_true", help="Push all 4 dataset variants with one command.")
    parser.add_argument(
        "--push-train-val-separately",
        action="store_true",
        help="Push train and validation as separate repos (<repo>-train / <repo>-validation).",
    )
    parser.add_argument(
        "--repo-id-train",
        default=None,
        help="Optional train repo id override when --push-train-val-separately is enabled.",
    )
    parser.add_argument(
        "--repo-id-validation",
        default=None,
        help="Optional validation repo id override when --push-train-val-separately is enabled.",
    )
    parser.add_argument(
        "--allow-duplicate-facts",
        action="store_true",
        help="Allow HF export/push to continue even when exact duplicate facts are detected in annotations.",
    )
    parser.add_argument(
        "--allow-ordering-issues",
        action="store_true",
        help="Allow HF export/push to continue when fact reading-order violations are detected in annotations.",
    )
    parser.add_argument(
        "--allow-format-issues",
        action="store_true",
        help="Allow HF export/push to continue when fact schema/date/value format issues are detected.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(".").resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Pass --token, export FINETREE_HF_TOKEN/HF_TOKEN, or configure HF_TOKEN in Doppler."
        )

    config_path = (root / args.config).resolve()
    cfg = load_finetune_config(config_path)
    export_dir = (root / args.export_dir).resolve()
    resize_bounds = _resolve_resize_bounds(args.min_pixels, args.max_pixels)
    resolved_min = resize_bounds[0] if resize_bounds else None
    resolved_max = resize_bounds[1] if resize_bounds else None
    instruction_mode = _normalize_instruction_mode(args.instruction_mode)
    excluded_doc_ids = _parse_excluded_doc_ids(args.exclude_doc_ids)
    main_repo_id = _resolve_repo_id(args.repo_id, token)
    compact_tokens = bool(args.compact_tokens or args.aggressive_compact_tokens)

    build_dataset(config_path, allow_format_issues=args.allow_format_issues)
    if instruction_mode == "source" or args.push_all_variants:
        assert_source_instruction_schema(root, fail_on_issues=True)
    assert_no_train_val_contamination(root)
    duplicate_report = assert_no_duplicate_facts(
        root,
        annotations_glob=cfg.data.annotations_glob,
        fail_on_duplicates=not args.allow_duplicate_facts,
    )
    if args.allow_duplicate_facts and int(duplicate_report["duplicate_rows"]) > 0:
        print(
            "WARNING: continuing with duplicate facts because --allow-duplicate-facts was set. "
            f"duplicate_rows={duplicate_report['duplicate_rows']}"
        )
    if cfg.data.fact_order_enforce:
        ordering_report = assert_fact_order(
            root,
            annotations_glob=cfg.data.annotations_glob,
            default_direction=cfg.data.fact_order_default_on_uncertain,
            row_tolerance_ratio=cfg.data.fact_order_row_tolerance_ratio,
            row_tolerance_min_px=cfg.data.fact_order_row_tolerance_min_px,
            fail_on_issues=not args.allow_ordering_issues,
        )
        if args.allow_ordering_issues and int(ordering_report["pages_with_order_issues"]) > 0:
            print(
                "WARNING: continuing with ordering issues because --allow-ordering-issues was set. "
                f"pages_with_order_issues={ordering_report['pages_with_order_issues']}"
            )
    else:
        print("FACT_ORDER_AUDIT: skipped (data.fact_order_enforce=false)")
    if cfg.data.fact_format_enforce:
        format_report = assert_fact_format(
            root,
            annotations_glob=cfg.data.annotations_glob,
            fail_on_issues=not args.allow_format_issues,
        )
        if args.allow_format_issues and int(format_report["facts_with_issues"]) > 0:
            print(
                "WARNING: continuing with format issues because --allow-format-issues was set. "
                f"facts_with_issues={format_report['facts_with_issues']}"
            )
    else:
        print("FACT_FORMAT_AUDIT: skipped (data.fact_format_enforce=false)")

    train_rows, val_rows = export_for_hf(
        root,
        export_dir,
        instruction_mode=instruction_mode,
        min_pixels=resolved_min,
        max_pixels=resolved_max,
        exclude_doc_ids=excluded_doc_ids,
        compact_tokens=compact_tokens,
        aggressive_compact_tokens=args.aggressive_compact_tokens,
    )
    dataset, _, _ = build_hf_dataset_from_export(export_dir)
    if args.push_train_val_separately:
        split_pushes = push_train_validation_separately(
            dataset,
            token=token,
            base_repo_id=main_repo_id,
            private=not args.public,
            repo_id_train=args.repo_id_train,
            repo_id_validation=args.repo_id_validation,
        )
        print(f"PUSHED_TRAIN_REPO: {split_pushes.get('train')}")
        print(f"PUSHED_VALIDATION_REPO: {split_pushes.get('validation')}")
    else:
        repo = push_to_hf(dataset, token=token, repo_id=main_repo_id, private=not args.public)
        print(f"PUSHED: {repo}")
    print(f"TRAIN_ROWS: {train_rows}")
    print(f"VAL_ROWS: {val_rows}")
    print(f"INSTRUCTION_MODE: {instruction_mode}")
    print(f"COMPACT_TOKENS: {compact_tokens}")
    print(f"AGGRESSIVE_COMPACT_TOKENS: {args.aggressive_compact_tokens}")
    print(f"EXPORT_DIR: {export_dir}")

    if args.push_all_variants:
        from . import push_dataset_hub_no_bbox as no_bbox_mod

        derived_minimal, derived_no_bbox, derived_no_bbox_min = _derive_variant_repo_ids(main_repo_id)
        repo_minimal = _resolve_repo_id(args.repo_id_minimal_instruction or derived_minimal, token)
        repo_no_bbox = _resolve_repo_id(args.repo_id_no_bbox or derived_no_bbox, token)
        repo_no_bbox_min = _resolve_repo_id(args.repo_id_no_bbox_minimal_instruction or derived_no_bbox_min, token)

        export_minimal = (root / "artifacts/hf_dataset_export_minimal_instruction").resolve()
        export_no_bbox = (root / "artifacts/hf_dataset_export_no_bbox").resolve()
        export_no_bbox_min = (root / "artifacts/hf_dataset_export_no_bbox_minimal_instruction").resolve()

        min_train, min_val = export_for_hf(
            root,
            export_minimal,
            instruction_mode="minimal",
            min_pixels=resolved_min,
            max_pixels=resolved_max,
            exclude_doc_ids=excluded_doc_ids,
            compact_tokens=compact_tokens,
            aggressive_compact_tokens=args.aggressive_compact_tokens,
        )
        min_ds, _, _ = build_hf_dataset_from_export(export_minimal)
        if args.push_train_val_separately:
            min_split_pushes = push_train_validation_separately(
                min_ds,
                token=token,
                base_repo_id=repo_minimal,
                private=not args.public,
            )
            print(f"PUSHED_MINIMAL_TRAIN_REPO: {min_split_pushes.get('train')}")
            print(f"PUSHED_MINIMAL_VALIDATION_REPO: {min_split_pushes.get('validation')}")
        else:
            pushed_minimal = push_to_hf(
                min_ds,
                token=token,
                repo_id=repo_minimal,
                private=not args.public,
            )
            print(f"PUSHED_MINIMAL: {pushed_minimal}")
        print(f"MINIMAL_TRAIN_ROWS: {min_train}")
        print(f"MINIMAL_VAL_ROWS: {min_val}")
        print(f"MINIMAL_EXPORT_DIR: {export_minimal}")

        nb_train, nb_val = no_bbox_mod.export_for_hf_no_bbox(
            root,
            export_no_bbox,
            instruction_mode="source",
            min_pixels=resolved_min,
            max_pixels=resolved_max,
            exclude_doc_ids=excluded_doc_ids,
            compact_tokens=compact_tokens,
            aggressive_compact_tokens=args.aggressive_compact_tokens,
        )
        nb_ds, _, _ = no_bbox_mod.build_hf_dataset_no_bbox_from_export(export_no_bbox, instruction_mode="source")
        if args.push_train_val_separately:
            nb_split_pushes = push_train_validation_separately(
                nb_ds,
                token=token,
                base_repo_id=repo_no_bbox,
                private=not args.public,
            )
            print(f"PUSHED_NO_BBOX_TRAIN_REPO: {nb_split_pushes.get('train')}")
            print(f"PUSHED_NO_BBOX_VALIDATION_REPO: {nb_split_pushes.get('validation')}")
        else:
            pushed_no_bbox = no_bbox_mod.push_to_hf_no_bbox(
                nb_ds,
                token=token,
                repo_id=repo_no_bbox,
                private=not args.public,
                instruction_mode="source",
            )
            print(f"PUSHED_NO_BBOX: {pushed_no_bbox}")
        print(f"NO_BBOX_TRAIN_ROWS: {nb_train}")
        print(f"NO_BBOX_VAL_ROWS: {nb_val}")
        print(f"NO_BBOX_EXPORT_DIR: {export_no_bbox}")

        nbm_train, nbm_val = no_bbox_mod.export_for_hf_no_bbox(
            root,
            export_no_bbox_min,
            instruction_mode="minimal",
            min_pixels=resolved_min,
            max_pixels=resolved_max,
            exclude_doc_ids=excluded_doc_ids,
            compact_tokens=compact_tokens,
            aggressive_compact_tokens=args.aggressive_compact_tokens,
        )
        nbm_ds, _, _ = no_bbox_mod.build_hf_dataset_no_bbox_from_export(export_no_bbox_min, instruction_mode="minimal")
        if args.push_train_val_separately:
            nbm_split_pushes = push_train_validation_separately(
                nbm_ds,
                token=token,
                base_repo_id=repo_no_bbox_min,
                private=not args.public,
            )
            print(f"PUSHED_NO_BBOX_MINIMAL_TRAIN_REPO: {nbm_split_pushes.get('train')}")
            print(f"PUSHED_NO_BBOX_MINIMAL_VALIDATION_REPO: {nbm_split_pushes.get('validation')}")
        else:
            pushed_no_bbox_min = no_bbox_mod.push_to_hf_no_bbox(
                nbm_ds,
                token=token,
                repo_id=repo_no_bbox_min,
                private=not args.public,
                instruction_mode="minimal",
            )
            print(f"PUSHED_NO_BBOX_MINIMAL: {pushed_no_bbox_min}")
        print(f"NO_BBOX_MINIMAL_TRAIN_ROWS: {nbm_train}")
        print(f"NO_BBOX_MINIMAL_VAL_ROWS: {nbm_val}")
        print(f"NO_BBOX_MINIMAL_EXPORT_DIR: {export_no_bbox_min}")
    return 0


__all__ = [
    "build_dataset",
    "assert_source_instruction_schema",
    "assert_no_train_val_contamination",
    "build_hf_dataset",
    "build_hf_dataset_from_export",
    "export_for_hf",
    "push_to_hf",
    "resolve_hf_token",
    "_dataset_for_push",
    "_resolve_repo_id",
    "_split_repo_ids",
    "push_train_validation_separately",
    "_resolve_resize_bounds",
    "_copy_or_resize_image",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
