from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pdf2image import convert_from_path, pdfinfo_from_path

from .annotation_core import PageState, bbox_to_list, default_page_meta, load_page_states, normalize_fact_data
from .page_issues import validate_document_issues
from .schema_io import load_any_schema

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class WorkspaceDocumentSummary:
    doc_id: str
    source_pdf: Optional[Path]
    images_dir: Path
    annotations_path: Path
    thumbnail_path: Optional[Path]
    page_count: int
    annotated_page_count: int
    progress_pct: int
    status: str
    updated_at: Optional[float]
    approved_page_count: int = 0
    annotated_token_count: int = 0
    fact_count: int = 0
    reg_flag_count: int = 0
    warning_count: int = 0
    pages_with_reg_flags: int = 0
    pages_with_warnings: int = 0
    checked: bool = False
    reviewed: bool = False


@dataclass(frozen=True)
class WorkspaceImportResult:
    document: WorkspaceDocumentSummary
    extraction_summary: dict[str, Any]
    copied_pdf: bool
    opened_existing: bool


def sanitize_doc_id(raw_name: str) -> str:
    text = unicodedata.normalize("NFC", str(raw_name or "").strip())
    cleaned = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "document"


def _legacy_sanitize_doc_id(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_name or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "document"


def raw_pdfs_dir(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "raw_pdfs"


def pdf_images_root(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "pdf_images"


def annotations_root(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "annotations"


def review_state_path(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "workspace_review_state.json"


def default_annotations_path(images_dir: Path, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return annotations_root(data_root) / f"{images_dir.name}.json"


def page_image_paths(images_dir: Path) -> list[Path]:
    if not Path(images_dir).is_dir():
        return []
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def missing_page_numbers(images_dir: Path, page_count: int) -> list[int]:
    missing: list[int] = []
    for idx in range(1, page_count + 1):
        target = Path(images_dir) / f"page_{idx:04d}.png"
        if not target.is_file():
            missing.append(idx)
    return missing


def build_page_ranges(page_numbers: list[int]) -> list[tuple[int, int]]:
    if not page_numbers:
        return []
    ranges: list[tuple[int, int]] = []
    start = page_numbers[0]
    prev = page_numbers[0]
    for page_num in page_numbers[1:]:
        if page_num == prev + 1:
            prev = page_num
            continue
        ranges.append((start, prev))
        start = page_num
        prev = page_num
    ranges.append((start, prev))
    return ranges


def ensure_pdf_images(pdf_path: Path, images_dir: Path, dpi: int = 200) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    if not pdf_path.is_file():
        raise RuntimeError(f"PDF not found: {pdf_path}")

    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf_info = pdfinfo_from_path(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF metadata for {pdf_path}: {exc}") from exc

    try:
        page_count = int((pdf_info or {}).get("Pages"))
    except Exception as exc:
        raise RuntimeError(f"Failed to determine page count for {pdf_path}.") from exc
    if page_count <= 0:
        raise RuntimeError(f"PDF has no pages: {pdf_path}")

    missing_pages = missing_page_numbers(images_dir, page_count)
    if not missing_pages:
        return {
            "action": "reused",
            "page_count": page_count,
            "created_pages": 0,
            "reused_pages": page_count,
            "missing_before": 0,
        }

    existing_before = page_count - len(missing_pages)
    created_pages = 0
    for first_page, last_page in build_page_ranges(missing_pages):
        try:
            rendered_pages = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                use_pdftocairo=True,
                first_page=first_page,
                last_page=last_page,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed converting pages {first_page}-{last_page} for {pdf_path}: {exc}"
            ) from exc

        expected = last_page - first_page + 1
        if len(rendered_pages) != expected:
            raise RuntimeError(
                f"Unexpected rendered page count for range {first_page}-{last_page}: "
                f"expected {expected}, got {len(rendered_pages)}"
            )

        for offset, page in enumerate(rendered_pages):
            page_num = first_page + offset
            target = images_dir / f"page_{page_num:04d}.png"
            page.save(target, format="PNG")
            created_pages += 1

    action = "created" if existing_before == 0 else "healed"
    return {
        "action": action,
        "page_count": page_count,
        "created_pages": created_pages,
        "reused_pages": page_count - created_pages,
        "missing_before": len(missing_pages),
    }


def load_annotation_payload(annotations_path: Path) -> dict[str, Any]:
    path = Path(annotations_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return load_any_schema(payload)


def load_reviewed_doc_ids(data_root: Path = DEFAULT_DATA_ROOT) -> set[str]:
    return load_workspace_check_state(data_root)[1]


def load_checked_doc_ids(data_root: Path = DEFAULT_DATA_ROOT) -> set[str]:
    return load_workspace_check_state(data_root)[0]


def load_workspace_check_state(data_root: Path = DEFAULT_DATA_ROOT) -> tuple[set[str], set[str]]:
    path = review_state_path(data_root)
    if not path.is_file():
        return set(), set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set(), set()
    checked = payload.get("checked_doc_ids") if isinstance(payload, dict) else None
    reviewed = payload.get("reviewed_doc_ids") if isinstance(payload, dict) else None
    checked_doc_ids = (
        {
            doc_id.strip()
            for doc_id in checked
            if isinstance(doc_id, str) and doc_id.strip()
        }
        if isinstance(checked, list)
        else set()
    )
    if not isinstance(reviewed, list):
        return checked_doc_ids, set()
    reviewed_doc_ids = {
        doc_id.strip()
        for doc_id in reviewed
        if isinstance(doc_id, str) and doc_id.strip()
    }
    return checked_doc_ids | reviewed_doc_ids, reviewed_doc_ids


def save_reviewed_doc_ids(reviewed_doc_ids: set[str], data_root: Path = DEFAULT_DATA_ROOT) -> None:
    checked_doc_ids, _ = load_workspace_check_state(data_root)
    save_workspace_check_state(checked_doc_ids | set(reviewed_doc_ids), reviewed_doc_ids, data_root=data_root)


def save_checked_doc_ids(checked_doc_ids: set[str], data_root: Path = DEFAULT_DATA_ROOT) -> None:
    _, reviewed_doc_ids = load_workspace_check_state(data_root)
    save_workspace_check_state(checked_doc_ids, reviewed_doc_ids & set(checked_doc_ids), data_root=data_root)


def save_workspace_check_state(
    checked_doc_ids: set[str],
    reviewed_doc_ids: set[str],
    data_root: Path = DEFAULT_DATA_ROOT,
) -> None:
    path = review_state_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_checked = {
        doc_id.strip() for doc_id in checked_doc_ids if isinstance(doc_id, str) and doc_id.strip()
    }
    normalized_reviewed = {
        doc_id.strip() for doc_id in reviewed_doc_ids if isinstance(doc_id, str) and doc_id.strip()
    }
    payload = {
        "checked_doc_ids": sorted(normalized_checked | normalized_reviewed),
        "reviewed_doc_ids": sorted(normalized_reviewed),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_document_checked(doc_id: str, checked: bool, data_root: Path = DEFAULT_DATA_ROOT) -> None:
    normalized = str(doc_id or "").strip()
    if not normalized:
        return
    checked_doc_ids, reviewed_doc_ids = load_workspace_check_state(data_root)
    if checked:
        checked_doc_ids.add(normalized)
    else:
        checked_doc_ids.discard(normalized)
        reviewed_doc_ids.discard(normalized)
    save_workspace_check_state(checked_doc_ids, reviewed_doc_ids, data_root=data_root)


def set_document_reviewed(doc_id: str, reviewed: bool, data_root: Path = DEFAULT_DATA_ROOT) -> None:
    normalized = str(doc_id or "").strip()
    if not normalized:
        return
    checked_doc_ids, reviewed_doc_ids = load_workspace_check_state(data_root)
    if reviewed:
        checked_doc_ids.add(normalized)
        reviewed_doc_ids.add(normalized)
    else:
        reviewed_doc_ids.discard(normalized)
    save_workspace_check_state(checked_doc_ids, reviewed_doc_ids, data_root=data_root)


def page_has_annotation(state: PageState, index: int) -> bool:
    _ = index
    meta = state.meta or {}

    title = meta.get("title")
    if isinstance(title, str):
        if title.strip():
            return True
    elif title not in (None, "", [], {}, False):
        return True

    return False


def page_is_approved(state: PageState) -> bool:
    meta = state.meta or {}
    status = meta.get("annotation_status")
    return isinstance(status, str) and status.strip().lower() == "approved"


def _state_map_for_document(images_dir: Path, annotations_path: Path) -> tuple[list[Path], dict[str, PageState]]:
    pages = page_image_paths(images_dir)
    payload = load_annotation_payload(annotations_path)
    states = load_page_states(payload, [page.name for page in pages]) if pages else {}
    return pages, states


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _page_annotation_payload(state: PageState, index: int) -> dict[str, Any]:
    meta = {**default_page_meta(index), **(state.meta or {})}
    facts = [
        {
            "bbox": bbox_to_list(box.bbox),
            **normalize_fact_data(box.fact),
        }
        for box in state.facts
    ]
    return {"meta": meta, "facts": facts}


def _estimate_annotation_tokens(payload: dict[str, Any]) -> int:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return len(_TOKEN_PATTERN.findall(text))


def compute_document_page_stats(images_dir: Path, annotations_path: Path) -> tuple[int, int, int, int]:
    pages, states = _state_map_for_document(images_dir, annotations_path)
    if not pages:
        return 0, 0, 0, 0
    annotated_pages = 0
    approved_pages = 0
    annotated_tokens = 0
    for idx, page in enumerate(pages):
        state = states.get(page.name, PageState(meta=default_page_meta(idx), facts=[]))
        if page_is_approved(state):
            approved_pages += 1
        if not page_has_annotation(state, idx):
            continue
        annotated_pages += 1
        annotated_tokens += _estimate_annotation_tokens(_page_annotation_payload(state, idx))
    return annotated_pages, approved_pages, len(pages), annotated_tokens


def compute_document_annotation_stats(images_dir: Path, annotations_path: Path) -> tuple[int, int, int]:
    annotated_pages, _approved_pages, page_count, annotated_tokens = compute_document_page_stats(
        images_dir,
        annotations_path,
    )
    return annotated_pages, page_count, annotated_tokens


def compute_document_progress(images_dir: Path, annotations_path: Path) -> tuple[int, int]:
    annotated_pages, page_count, _annotated_tokens = compute_document_annotation_stats(images_dir, annotations_path)
    return annotated_pages, page_count


def compute_document_fact_count(images_dir: Path, annotations_path: Path) -> int:
    pages, states = _state_map_for_document(images_dir, annotations_path)
    if not pages:
        return 0
    fact_count = 0
    for idx, page in enumerate(pages):
        state = states.get(page.name, PageState(meta=default_page_meta(idx), facts=[]))
        fact_count += len(state.facts)
    return fact_count


def compute_document_issue_summary(images_dir: Path, annotations_path: Path):
    pages, states = _state_map_for_document(images_dir, annotations_path)
    ordered_states = []
    for idx, page in enumerate(pages):
        ordered_states.append((page.name, states.get(page.name, PageState(meta=default_page_meta(idx), facts=[]))))
    return validate_document_issues(ordered_states)


def _latest_mtime(*paths: Optional[Path]) -> Optional[float]:
    mtimes: list[float] = []
    for path in paths:
        if path is None:
            continue
        try:
            if Path(path).exists():
                mtimes.append(Path(path).stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else None


def _resolve_doc_path_aliases(root: Path, *, doc_id: str, suffix: Optional[str] = None, directories_only: bool = False) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []

    legacy_doc_id = _legacy_sanitize_doc_id(doc_id)
    candidates: list[Path] = []
    if directories_only:
        candidates = [path for path in root.iterdir() if path.is_dir()]
        exact_matches = [path for path in candidates if path.name == doc_id]
        alias_matches = [
            path
            for path in candidates
            if path.name != doc_id
            and (
                sanitize_doc_id(path.name) == doc_id
                or (
                    doc_id == legacy_doc_id
                    and _legacy_sanitize_doc_id(path.name) == legacy_doc_id
                )
            )
        ]
    else:
        normalized_suffix = str(suffix or "").lower()
        candidates = [path for path in root.iterdir() if path.is_file() and (not normalized_suffix or path.suffix.lower() == normalized_suffix)]
        exact_matches = [path for path in candidates if path.stem == doc_id]
        alias_matches = [
            path
            for path in candidates
            if path.stem != doc_id
            and (
                sanitize_doc_id(path.stem) == doc_id
                or (
                    doc_id == legacy_doc_id
                    and _legacy_sanitize_doc_id(path.stem) == legacy_doc_id
                )
            )
        ]

    if exact_matches:
        return sorted(exact_matches)
    return sorted(alias_matches)


def _resolve_images_dir(doc_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    matches = _resolve_doc_path_aliases(pdf_images_root(data_root), doc_id=doc_id, directories_only=True)
    if matches:
        return matches[0]
    return pdf_images_root(data_root) / doc_id


def _resolve_annotations_path(doc_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    matches = _resolve_doc_path_aliases(annotations_root(data_root), doc_id=doc_id, suffix=".json")
    if matches:
        return matches[0]
    return annotations_root(data_root) / f"{doc_id}.json"


def _resolve_source_pdf(doc_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    matches = _resolve_doc_path_aliases(raw_pdfs_dir(data_root), doc_id=doc_id, suffix=".pdf")
    if matches:
        return matches[0]
    return raw_pdfs_dir(data_root) / f"{doc_id}.pdf"


def build_document_summary(
    doc_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    checked_doc_ids: Optional[set[str]] = None,
    reviewed_doc_ids: Optional[set[str]] = None,
) -> WorkspaceDocumentSummary:
    data_root = Path(data_root)
    images_dir = _resolve_images_dir(doc_id, data_root=data_root)
    annotations_path = _resolve_annotations_path(doc_id, data_root=data_root)
    source_pdf = _resolve_source_pdf(doc_id, data_root=data_root)
    if checked_doc_ids is None or reviewed_doc_ids is None:
        loaded_checked_ids, loaded_reviewed_ids = load_workspace_check_state(data_root)
        checked_ids = checked_doc_ids if checked_doc_ids is not None else loaded_checked_ids
        reviewed_ids = reviewed_doc_ids if reviewed_doc_ids is not None else loaded_reviewed_ids
    else:
        checked_ids = checked_doc_ids
        reviewed_ids = reviewed_doc_ids
    page_paths = page_image_paths(images_dir)
    annotated_pages, approved_pages, page_count, annotated_tokens = compute_document_page_stats(
        images_dir,
        annotations_path,
    )
    fact_count = compute_document_fact_count(images_dir, annotations_path)
    issue_summary = compute_document_issue_summary(images_dir, annotations_path)
    can_be_checked = page_count > 0 and annotated_pages >= page_count
    thumbnail_path = page_paths[0] if page_paths else None
    progress_pct = int(round((annotated_pages / page_count) * 100)) if page_count else 0
    if page_count == 0 and source_pdf.is_file():
        status = "Needs extraction"
    elif page_count == 0:
        status = "Missing pages"
    elif progress_pct >= 100:
        status = "Complete"
    elif annotated_pages > 0:
        status = "In progress"
    else:
        status = "Ready"
    updated_at = _latest_mtime(source_pdf if source_pdf.is_file() else None, annotations_path, thumbnail_path)
    return WorkspaceDocumentSummary(
        doc_id=doc_id,
        source_pdf=source_pdf if source_pdf.is_file() else None,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=thumbnail_path,
        page_count=page_count,
        annotated_page_count=annotated_pages,
        approved_page_count=approved_pages,
        annotated_token_count=annotated_tokens,
        fact_count=fact_count,
        progress_pct=progress_pct,
        status=status,
        updated_at=updated_at,
        reg_flag_count=issue_summary.reg_flag_count,
        warning_count=issue_summary.warning_count,
        pages_with_reg_flags=issue_summary.pages_with_reg_flags,
        pages_with_warnings=issue_summary.pages_with_warnings,
        checked=((doc_id in checked_ids) and can_be_checked),
        reviewed=((doc_id in reviewed_ids) and (doc_id in checked_ids) and can_be_checked and issue_summary.reg_flag_count == 0),
    )


def discover_workspace_documents(data_root: Path = DEFAULT_DATA_ROOT) -> list[WorkspaceDocumentSummary]:
    data_root = Path(data_root)
    doc_ids: set[str] = set()
    checked_doc_ids, reviewed_doc_ids = load_workspace_check_state(data_root)

    images_root = pdf_images_root(data_root)
    image_doc_ids: set[str] = set()
    if images_root.is_dir():
        image_doc_ids = {sanitize_doc_id(path.name) for path in images_root.iterdir() if path.is_dir()}
        doc_ids.update(image_doc_ids)

    ann_root = annotations_root(data_root)
    annotation_doc_ids: set[str] = set()
    if ann_root.is_dir():
        annotation_doc_ids = {sanitize_doc_id(path.stem) for path in ann_root.iterdir() if path.is_file() and path.suffix.lower() == ".json"}
        doc_ids.update(annotation_doc_ids)

    raw_root = raw_pdfs_dir(data_root)
    if raw_root.is_dir():
        raw_pdfs = [path for path in raw_root.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"]
        raw_legacy_counts = Counter(_legacy_sanitize_doc_id(path.stem) for path in raw_pdfs)
        known_legacy_doc_ids = {_legacy_sanitize_doc_id(doc_id) for doc_id in (image_doc_ids | annotation_doc_ids)}
        for path in raw_pdfs:
            new_doc_id = sanitize_doc_id(path.stem)
            legacy_doc_id = _legacy_sanitize_doc_id(path.stem)
            # Backward compatibility: keep legacy ASCII id only when unambiguous.
            if (
                new_doc_id != legacy_doc_id
                and raw_legacy_counts[legacy_doc_id] == 1
                and legacy_doc_id in known_legacy_doc_ids
                and new_doc_id not in image_doc_ids
                and new_doc_id not in annotation_doc_ids
            ):
                doc_ids.add(legacy_doc_id)
            else:
                doc_ids.add(new_doc_id)

    documents = [
        build_document_summary(
            doc_id,
            data_root=data_root,
            checked_doc_ids=checked_doc_ids,
            reviewed_doc_ids=reviewed_doc_ids,
        )
        for doc_id in sorted(doc_ids)
    ]
    documents.sort(key=lambda doc: (doc.updated_at or 0.0, doc.doc_id), reverse=True)
    return documents


def import_pdf_to_workspace(source_pdf: Path, data_root: Path = DEFAULT_DATA_ROOT, dpi: int = 200) -> WorkspaceImportResult:
    source_pdf = Path(source_pdf).expanduser().resolve()
    if not source_pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {source_pdf}")
    if source_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {source_pdf}")

    data_root = Path(data_root)
    doc_id = sanitize_doc_id(source_pdf.stem)
    managed_pdf = raw_pdfs_dir(data_root) / f"{doc_id}.pdf"
    images_dir = pdf_images_root(data_root) / doc_id
    managed_pdf.parent.mkdir(parents=True, exist_ok=True)
    images_dir.parent.mkdir(parents=True, exist_ok=True)
    annotations_root(data_root).mkdir(parents=True, exist_ok=True)

    copied_pdf = False
    opened_existing = managed_pdf.exists()
    try:
        same_file = managed_pdf.exists() and managed_pdf.samefile(source_pdf)
    except OSError:
        same_file = False
    if not managed_pdf.exists():
        shutil.copy2(source_pdf, managed_pdf)
        copied_pdf = True
        opened_existing = False
    elif same_file:
        opened_existing = False

    extraction_summary = ensure_pdf_images(managed_pdf, images_dir, dpi=dpi)
    document = build_document_summary(doc_id, data_root=data_root)
    return WorkspaceImportResult(
        document=document,
        extraction_summary=extraction_summary,
        copied_pdf=copied_pdf,
        opened_existing=opened_existing,
    )
