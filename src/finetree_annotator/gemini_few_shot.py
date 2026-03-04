from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

DEFAULT_TEST_FEW_SHOT_PAGES: tuple[str, ...] = (
    "page_0001.png",
    "page_0004.png",
    "page_0009.png",
    "page_0002.png",
)
DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS: tuple[tuple[str, str], ...] = (
    ("test", "page_0009.png"),
    ("test", "page_0004.png"),
    ("test", "page_0005.png"),
    ("test", "page_0010.png"),
    ("pdf_3", "page_0018.png"),
    ("pdf_3", "page_0023.png"),
    ("pdf_2", "page_0008.png"),
)


def build_repo_roots(*, cwd: Optional[Path] = None, anchor_file: Optional[Path] = None) -> list[Path]:
    roots: list[Path] = []
    base_cwd = (cwd or Path.cwd()).resolve()
    roots.append(base_cwd)

    anchor = anchor_file or Path(__file__).resolve()
    roots.extend(anchor.resolve().parents)

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def resolve_repo_relative_path(relative_path: str, *, repo_roots: Sequence[Path]) -> Optional[Path]:
    rel = str(relative_path or "").strip()
    if not rel:
        return None
    for root in repo_roots:
        candidate = (root / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def _page_payload_map(pages: Iterable[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_name = str(page.get("image") or "").strip()
        if not image_name:
            continue
        out[image_name] = page
    return out


def _expected_payload_from_page(page_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": page_payload.get("meta") if isinstance(page_payload.get("meta"), dict) else {},
        "facts": page_payload.get("facts") if isinstance(page_payload.get("facts"), list) else [],
    }


def _load_annotations_payload(
    *,
    relative_path: str,
    repo_roots: Sequence[Path],
    warnings: list[str],
) -> Optional[dict[str, Any]]:
    annotations_path = resolve_repo_relative_path(relative_path, repo_roots=repo_roots)
    if annotations_path is None:
        warnings.append(f"Few-shot annotations file missing: {relative_path}")
        return None
    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Failed to parse {relative_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"Few-shot annotations JSON is not an object: {relative_path}")
        return None
    return payload


def load_test_pdf_few_shot_examples(
    *,
    repo_roots: Sequence[Path],
    page_names: Sequence[str] = DEFAULT_TEST_FEW_SHOT_PAGES,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    examples: list[dict[str, Any]] = []

    payload = _load_annotations_payload(
        relative_path="data/annotations/test.json",
        repo_roots=repo_roots,
        warnings=warnings,
    )
    if payload is None:
        return examples, warnings

    pages = payload.get("pages")
    if not isinstance(pages, list):
        warnings.append("Few-shot annotations JSON has no valid pages list.")
        return examples, warnings
    page_map = _page_payload_map(pages)

    for page_name in page_names:
        image_path = resolve_repo_relative_path(
            f"data/pdf_images/test/{page_name}",
            repo_roots=repo_roots,
        )
        if image_path is None or not image_path.is_file():
            warnings.append(f"Few-shot image missing: data/pdf_images/test/{page_name}")
            continue

        page_payload = page_map.get(page_name)
        if page_payload is None:
            warnings.append(f"Few-shot annotation page missing in test.json: {page_name}")
            continue

        examples.append(
            {
                "image_path": image_path,
                "expected_json": json.dumps(_expected_payload_from_page(page_payload), ensure_ascii=False, separators=(",", ":")),
            }
        )

    return examples, warnings


def load_complex_few_shot_examples(
    *,
    repo_roots: Sequence[Path],
    selections: Sequence[tuple[str, str]] = DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    examples: list[dict[str, Any]] = []
    dataset_cache: dict[str, tuple[dict[str, dict[str, Any]], str]] = {}

    for raw_dataset_name, raw_page_name in selections:
        dataset_name = str(raw_dataset_name or "").strip()
        page_name = str(raw_page_name or "").strip()
        if not dataset_name or not page_name:
            continue

        cached = dataset_cache.get(dataset_name)
        if cached is None:
            annotations_rel = f"data/annotations/{dataset_name}.json"
            payload = _load_annotations_payload(
                relative_path=annotations_rel,
                repo_roots=repo_roots,
                warnings=warnings,
            )
            if payload is None:
                continue

            pages = payload.get("pages")
            if not isinstance(pages, list):
                warnings.append(f"Few-shot annotations JSON has no valid pages list: {annotations_rel}")
                continue
            page_map = _page_payload_map(pages)
            if not page_map:
                warnings.append(f"Few-shot annotations has no importable pages: {annotations_rel}")
                continue

            image_dir = str(payload.get("images_dir") or "").strip() or f"data/pdf_images/{dataset_name}"
            cached = (page_map, image_dir)
            dataset_cache[dataset_name] = cached

        page_map, image_dir = cached
        image_rel = image_dir.rstrip("/") + "/" + page_name
        image_path = resolve_repo_relative_path(image_rel, repo_roots=repo_roots)
        if image_path is None or not image_path.is_file():
            warnings.append(f"Few-shot image missing: {image_rel}")
            continue

        page_payload = page_map.get(page_name)
        if page_payload is None:
            warnings.append(f"Few-shot annotation page missing in {dataset_name}.json: {page_name}")
            continue

        examples.append(
            {
                "image_path": image_path,
                "expected_json": json.dumps(_expected_payload_from_page(page_payload), ensure_ascii=False, separators=(",", ":")),
            }
        )

    return examples, warnings


__all__ = [
    "DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS",
    "DEFAULT_TEST_FEW_SHOT_PAGES",
    "build_repo_roots",
    "load_complex_few_shot_examples",
    "load_test_pdf_few_shot_examples",
    "resolve_repo_relative_path",
]
