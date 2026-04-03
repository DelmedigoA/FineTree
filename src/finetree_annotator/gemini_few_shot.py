from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .annotation_core import PageState, build_annotations_payload_with_findings, default_page_meta
from .bbox_utils import bbox_to_list
from .fact_normalization import normalize_fact_payload
from .schema_contract import PROMPT_FACT_KEYS, PROMPT_PAGE_META_KEYS
from .schemas import PageMeta
from .workspace import page_is_approved

DEFAULT_TEST_FEW_SHOT_PAGES: tuple[str, ...] = (
    "page_0001.png",
    "page_0004.png",
    "page_0009.png",
    "page_0002.png",
)
DEFAULT_TEST_ONE_SHOT_PAGE = "page_0005.png"
DEFAULT_2015_TWO_SHOT_SELECTIONS: tuple[tuple[str, str], ...] = (
    ("2015", "page_0004.png"),
    ("2015", "page_0011.png"),
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
_CUSTOM_PAGE_SPEC_RE = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*$")


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


def _prompt_page_meta_payload(raw_meta: Any) -> dict[str, Any]:
    validated = PageMeta.model_validate(raw_meta if isinstance(raw_meta, dict) else {}).model_dump(mode="json")
    return {key: validated.get(key) for key in PROMPT_PAGE_META_KEYS}


def _prompt_fact_payload(raw_fact: Any) -> dict[str, Any]:
    normalized, _warnings = normalize_fact_payload(raw_fact if isinstance(raw_fact, dict) else {}, include_bbox=True)
    bbox_value = raw_fact.get("bbox") if isinstance(raw_fact, dict) else normalized.get("bbox")
    payload = {"bbox": bbox_to_list(bbox_value)}
    payload.update({key: normalized.get(key) for key in PROMPT_FACT_KEYS})
    return payload


def _expected_payload_from_page(page_payload: dict[str, Any]) -> dict[str, Any]:
    facts = page_payload.get("facts") if isinstance(page_payload.get("facts"), list) else []
    return {
        "pages": [
            {
                "image": str(page_payload.get("image") or "").strip() or None,
                "meta": _prompt_page_meta_payload(page_payload.get("meta")),
                "facts": [_prompt_fact_payload(fact) for fact in facts if isinstance(fact, dict)],
            }
        ]
    }


def _enrich_input_fact_payload(fact_num: int, raw_fact: dict[str, Any]) -> dict[str, Any]:
    """Strips fact to {fact_num, bbox, value} for use as enrich few-shot input."""
    normalized, _ = normalize_fact_payload(raw_fact, include_bbox=True)
    bbox_value = raw_fact.get("bbox") if isinstance(raw_fact, dict) else normalized.get("bbox")
    return {
        "fact_num": fact_num,
        "bbox": bbox_to_list(bbox_value),
        "value": normalized.get("value"),
    }


def _enrich_expected_fact_payload(fact_num: int, raw_fact: dict[str, Any]) -> dict[str, Any]:
    """Builds full expected output fact for enrich few-shots (all fields, preserving bbox/value)."""
    normalized, _ = normalize_fact_payload(raw_fact, include_bbox=True)
    bbox_value = raw_fact.get("bbox") if isinstance(raw_fact, dict) else normalized.get("bbox")
    payload: dict[str, Any] = {
        "fact_num": fact_num,
        "bbox": bbox_to_list(bbox_value),
    }
    payload.update({key: normalized.get(key) for key in PROMPT_FACT_KEYS})
    return payload


def _enrich_payloads_from_page(
    page_payload: dict[str, Any],
    *,
    reading_direction: str = "rtl",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (input_payload, expected_payload) for a page in enrich few-shot format.

    Input facts contain only {fact_num, bbox, value}; expected facts contain all fields.
    Both are ordered by RTL/LTR geometry and re-numbered sequentially.
    """
    raw_facts = page_payload.get("facts") if isinstance(page_payload.get("facts"), list) else []
    valid_facts = [f for f in raw_facts if isinstance(f, dict)]

    # Order by geometry using canonical reading order
    try:
        from .fact_ordering import canonical_fact_order_indices
        indices = canonical_fact_order_indices(valid_facts, direction=reading_direction)  # type: ignore[arg-type]
        ordered_facts = [valid_facts[i] for i in indices]
    except Exception:
        ordered_facts = valid_facts

    meta = _prompt_page_meta_payload(page_payload.get("meta"))
    image = str(page_payload.get("image") or "").strip() or None

    input_facts = [_enrich_input_fact_payload(i + 1, f) for i, f in enumerate(ordered_facts)]
    expected_facts = [_enrich_expected_fact_payload(i + 1, f) for i, f in enumerate(ordered_facts)]

    input_payload: dict[str, Any] = {"pages": [{"image": image, "meta": meta, "facts": input_facts}]}
    expected_payload: dict[str, Any] = {"pages": [{"image": image, "meta": meta, "facts": expected_facts}]}
    return input_payload, expected_payload


def load_auto_annotate_enrich_few_shot_examples(
    *,
    repo_roots: Sequence[Path],
    selections: Sequence[tuple[str, str]] = DEFAULT_2015_TWO_SHOT_SELECTIONS,
    reading_direction: str = "rtl",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load few-shot examples for Auto-Annotate enrich mode.

    Each example dict contains:
      - image_path: Path to the example image
      - input_json: JSON string with {fact_num, bbox, value} per fact (seed input)
      - expected_json: JSON string with full annotation per fact (target output)
    """
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

        input_payload, expected_payload = _enrich_payloads_from_page(
            page_payload, reading_direction=reading_direction
        )
        examples.append(
            {
                "image_path": image_path,
                "input_json": json.dumps(input_payload, ensure_ascii=False, separators=(",", ":")),
                "expected_json": json.dumps(expected_payload, ensure_ascii=False, separators=(",", ":")),
            }
        )

    return examples, warnings


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


def parse_document_few_shot_page_spec(
    page_spec: str,
    *,
    page_count: int,
    current_page_index: int,
) -> tuple[list[int], str | None]:
    raw_spec = str(page_spec or "").strip()
    if not raw_spec:
        return [], "Enter a custom page like `3` or `1-3`."
    match = _CUSTOM_PAGE_SPEC_RE.fullmatch(raw_spec)
    if match is None:
        return [], "Custom pages must be a single page like `3` or a range like `1-3`."

    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start <= 0 or end <= 0:
        return [], "Custom pages must be 1-based page numbers greater than 0."
    if end < start:
        return [], "Custom page ranges must be ascending, for example `1-3`."
    if start > page_count or end > page_count:
        return [], f"Custom pages must be within this document's range: 1-{page_count}."

    selected_indices = list(range(start - 1, end))
    if current_page_index in selected_indices:
        current_page_num = current_page_index + 1
        return [], f"Custom few-shot pages cannot include the current page ({current_page_num})."
    return selected_indices, None


def load_document_few_shot_examples(
    *,
    images_dir: Path,
    page_images: Sequence[Path],
    page_states: dict[str, PageState],
    current_page_index: int,
    source: str,
    previous_count: int = 2,
    page_spec: str = "",
    document_meta: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    page_count = len(page_images)
    if page_count <= 0:
        return [], [], "Load a document before using document-based few-shot examples."
    if current_page_index < 0 or current_page_index >= page_count:
        return [], [], "The current page is unavailable for document-based few-shot examples."

    selected_indices: list[int] = []
    warnings: list[str] = []

    if source == "previous_pages":
        requested = max(int(previous_count), 0)
        if requested < 1:
            return [], [], "Previous pages count must be at least 1."
        selected_desc: list[int] = []
        for page_index in range(current_page_index - 1, -1, -1):
            page_name = page_images[page_index].name
            state = page_states.get(page_name, PageState(meta=default_page_meta(page_index), facts=[]))
            if not page_is_approved(state):
                continue
            selected_desc.append(page_index)
            if len(selected_desc) >= requested:
                break
        if not selected_desc:
            return [], [], "No approved previous pages are available for few-shot."
        selected_indices = list(reversed(selected_desc))
        if len(selected_indices) < requested:
            warnings.append(
                f"Requested {requested} previous approved page(s), found {len(selected_indices)}."
            )
    elif source == "custom_pages":
        selected_indices, error_message = parse_document_few_shot_page_spec(
            page_spec,
            page_count=page_count,
            current_page_index=current_page_index,
        )
        if error_message:
            return [], [], error_message
        approved_indices: list[int] = []
        skipped_pages: list[int] = []
        for page_index in selected_indices:
            page_name = page_images[page_index].name
            state = page_states.get(page_name, PageState(meta=default_page_meta(page_index), facts=[]))
            if page_is_approved(state):
                approved_indices.append(page_index)
            else:
                skipped_pages.append(page_index + 1)
        if not approved_indices:
            return [], [], "No approved pages were found in the custom few-shot selection."
        if skipped_pages:
            skipped_text = ", ".join(str(page_num) for page_num in skipped_pages)
            warnings.append(f"Skipped unapproved custom page(s): {skipped_text}.")
        selected_indices = approved_indices
    else:
        return [], [], f"Unsupported document few-shot source: {source}"

    payload, _findings = build_annotations_payload_with_findings(
        Path(images_dir),
        page_images,
        page_states,
        document_meta=document_meta,
    )
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return [], [], "Live annotations payload is missing pages for few-shot examples."
    page_payload_map = _page_payload_map(raw_pages)

    examples: list[dict[str, Any]] = []
    for page_index in selected_indices:
        page_path = page_images[page_index]
        page_payload = page_payload_map.get(page_path.name)
        if page_payload is None:
            warnings.append(f"Few-shot page payload missing for page {page_index + 1}.")
            continue
        examples.append(
            {
                "image_path": page_path.resolve(),
                "expected_json": json.dumps(
                    _expected_payload_from_page(page_payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    if not examples:
        return [], warnings, "No eligible few-shot examples could be prepared from the current document."
    return examples, warnings, None


__all__ = [
    "DEFAULT_2015_TWO_SHOT_SELECTIONS",
    "DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS",
    "DEFAULT_TEST_ONE_SHOT_PAGE",
    "DEFAULT_TEST_FEW_SHOT_PAGES",
    "build_repo_roots",
    "load_auto_annotate_enrich_few_shot_examples",
    "load_complex_few_shot_examples",
    "load_document_few_shot_examples",
    "load_test_pdf_few_shot_examples",
    "parse_document_few_shot_page_spec",
    "resolve_repo_relative_path",
]
