"""Dataset version management and HuggingFace push API."""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .deps import get_data_root
from .sse_helpers import sse_event, sse_response
from ..schema_contract import (
    PROMPT_FACT_KEYS,
    PROMPT_PAGE_META_KEYS,
    REQUIRED_PROMPT_CANONICAL_KEYS,
    build_custom_extraction_prompt_template,
)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

SplitName = Literal["train", "test", "val", "exclude"]
PushStatus = Literal["never", "pushed"]
_TRACKED_SPLITS: tuple[SplitName, ...] = ("train", "test", "val", "exclude")

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise financial statement extraction system. "
    "Return only valid JSON that matches the required schema."
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ExportConfig(BaseModel):
    hf_repo: str = ""
    push_mode: Literal["single", "separate"] = "single"
    image_scaling: bool = True
    max_pixels: int = 1_400_000
    min_pixels: int | None = None
    bbox_grid_norm: bool = True
    values_norm: bool = True
    compact_mode: Literal["raw", "compact", "aggressive"] = "compact"
    drop_date: bool = False
    selected_fact_keys: list[str] = Field(default_factory=lambda: list(PROMPT_FACT_KEYS))
    selected_page_meta_keys: list[str] = Field(default_factory=lambda: list(PROMPT_PAGE_META_KEYS))
    include_bbox: bool = True
    approved_pages_only: bool = True


class SplitStatsEntry(BaseModel):
    doc_count: int = 0
    page_count: int = 0


class DatasetVersion(BaseModel):
    version_id: str
    name: str
    created_at: float
    updated_at: float | None = None
    split_assignments: dict[str, SplitName] = Field(default_factory=dict)
    export_config: ExportConfig = Field(default_factory=ExportConfig)
    split_stats: dict[str, SplitStatsEntry] = Field(default_factory=dict)
    push_status: PushStatus = "never"
    last_pushed_at: float | None = None
    pushed_repos: dict[str, str] = Field(default_factory=dict)


class CreateVersionRequest(BaseModel):
    name: str
    split_assignments: dict[str, SplitName] = Field(default_factory=dict)
    export_config: ExportConfig = Field(default_factory=ExportConfig)


class UpdateVersionRequest(BaseModel):
    name: str | None = None
    split_assignments: dict[str, SplitName] | None = None
    export_config: ExportConfig | None = None


class PreviewRequest(BaseModel):
    n_per_split: int = 2


class StatelessPreviewRequest(BaseModel):
    split: str = "train"
    assignments: dict[str, SplitName] = Field(default_factory=dict)
    export_config: ExportConfig = Field(default_factory=ExportConfig)
    n_per_split: int = 2


class StatelessPushRequest(BaseModel):
    assignments: dict[str, SplitName] = Field(default_factory=dict)
    export_config: ExportConfig = Field(default_factory=ExportConfig)


class TokenStatsRequest(BaseModel):
    tokenizer: str = "Qwen/Qwen3.5-27B"


class StatelessTokenStatsRequest(BaseModel):
    tokenizer: str = "Qwen/Qwen3.5-27B"
    assignments: dict[str, SplitName] = Field(default_factory=dict)
    export_config: ExportConfig = Field(default_factory=ExportConfig)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _versions_dir(data_root: Path) -> Path:
    d = data_root / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _version_path(data_root: Path, version_id: str) -> Path:
    return _versions_dir(data_root) / f"{version_id}.json"


def _empty_split_stats() -> dict[str, SplitStatsEntry]:
    return {split: SplitStatsEntry() for split in _TRACKED_SPLITS}


def _count_exportable_pages_for_doc(
    data_root: Path,
    doc_id: str,
    *,
    approved_pages_only: bool,
) -> int:
    ann_path = data_root / "annotations" / f"{doc_id}.json"
    if not ann_path.is_file():
        return 0
    try:
        from ..schema_io import load_any_schema

        payload = load_any_schema(json.loads(ann_path.read_text("utf-8")))
    except Exception:
        return 0

    pages = payload.get("pages") or []
    if not isinstance(pages, list):
        return 0

    count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        meta = page.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        status = str(meta.get("annotation_status") or "").strip().lower()
        if approved_pages_only and status != "approved":
            continue
        image_name = str(page.get("image") or "").strip()
        if not image_name:
            continue
        image_path = data_root / "pdf_images" / doc_id / image_name
        if not image_path.is_file():
            continue
        count += 1
    return count


def _compute_split_stats(
    data_root: Path,
    split_assignments: dict[str, SplitName],
    config: ExportConfig,
) -> dict[str, SplitStatsEntry]:
    stats = _empty_split_stats()
    for doc_id, split in split_assignments.items():
        if split not in stats:
            continue
        stats[split].doc_count += 1
        if split == "exclude":
            continue
        stats[split].page_count += _count_exportable_pages_for_doc(
            data_root,
            doc_id,
            approved_pages_only=bool(config.approved_pages_only),
        )
    return stats


def _normalize_split_stats(
    data_root: Path,
    version: DatasetVersion,
) -> tuple[dict[str, SplitStatsEntry], bool]:
    changed = False
    stats = _empty_split_stats()
    raw_stats = version.split_stats or {}
    for split, entry in raw_stats.items():
        if split not in stats:
            changed = True
            continue
        if isinstance(entry, SplitStatsEntry):
            stats[split] = entry
        else:
            try:
                stats[split] = SplitStatsEntry.model_validate(entry)
            except Exception:
                changed = True
    if not raw_stats:
        stats = _compute_split_stats(data_root, version.split_assignments, version.export_config)
        changed = True
    return stats, changed


def _hydrate_version(data_root: Path, version: DatasetVersion, *, persist: bool) -> DatasetVersion:
    changed = False
    if version.updated_at is None:
        version.updated_at = version.created_at
        changed = True
    split_stats, split_stats_changed = _normalize_split_stats(data_root, version)
    if split_stats_changed or version.split_stats != split_stats:
        version.split_stats = split_stats
        changed = True
    if version.push_status == "never" and version.last_pushed_at and version.pushed_repos:
        version.push_status = "pushed"
        changed = True
    if version.push_status == "pushed" and not version.last_pushed_at:
        version.push_status = "never"
        version.pushed_repos = {}
        changed = True
    if persist and changed:
        _save_version(data_root, version)
    return version


def _load_version(data_root: Path, version_id: str) -> DatasetVersion:
    path = _version_path(data_root, version_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")
    raw = json.loads(path.read_text("utf-8"))
    version = DatasetVersion.model_validate(raw)
    return _hydrate_version(data_root, version, persist=True)


def _save_version(data_root: Path, version: DatasetVersion) -> None:
    path = _version_path(data_root, version.version_id)
    path.write_text(version.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def _build_split_rows(
    data_root: Path,
    doc_ids: list[str],
    split: str,
    config: ExportConfig,
) -> list[dict[str, Any]]:
    """Build chat-format JSONL rows for a list of doc_ids."""
    from ..schema_io import load_any_schema
    from ..fact_normalization import normalize_fact_payload

    instruction = build_custom_extraction_prompt_template(
        fact_keys=config.selected_fact_keys,
        page_meta_keys=config.selected_page_meta_keys,
        include_bbox=config.include_bbox,
    )

    rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        ann_path = data_root / "annotations" / f"{doc_id}.json"
        if not ann_path.is_file():
            continue
        payload = load_any_schema(json.loads(ann_path.read_text("utf-8")))
        pages = payload.get("pages", [])
        if not isinstance(pages, list):
            continue
        for page in pages:
            if not isinstance(page, dict):
                continue
            meta = page.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            status = str(meta.get("annotation_status") or "").lower()
            if config.approved_pages_only and status != "approved":
                continue
            image_name = str(page.get("image") or "").strip()
            if not image_name:
                continue
            image_path = data_root / "pdf_images" / doc_id / image_name
            if not image_path.is_file():
                continue

            # Build filtered page data.
            filtered_meta = {k: meta.get(k) for k in config.selected_page_meta_keys}
            facts = page.get("facts") or []
            filtered_facts: list[dict[str, Any]] = []
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                if config.values_norm:
                    fact, _ = normalize_fact_payload(fact, include_bbox=("bbox" in fact))
                filtered_fact = {
                    k: v
                    for k, v in fact.items()
                    if k == "bbox" or k in config.selected_fact_keys
                }
                if not config.include_bbox:
                    filtered_fact.pop("bbox", None)
                if config.drop_date:
                    filtered_fact.pop("date", None)
                filtered_facts.append(filtered_fact)

            page_payload = {"meta": filtered_meta, "facts": filtered_facts}
            assistant_text = json.dumps(page_payload, ensure_ascii=False)

            rows.append(
                {
                    "split": split,
                    "doc_id": doc_id,
                    "image": str(image_path),
                    "system": _DEFAULT_SYSTEM_PROMPT,
                    "instruction": instruction,
                    "text": assistant_text,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Endpoints — version CRUD
# ---------------------------------------------------------------------------


@router.get("/versions")
def list_versions() -> list[dict[str, Any]]:
    """List all dataset versions."""
    data_root = get_data_root()
    versions_dir = _versions_dir(data_root)
    result: list[DatasetVersion] = []
    for path in sorted(versions_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text("utf-8"))
            version = DatasetVersion.model_validate(raw)
            result.append(_hydrate_version(data_root, version, persist=True))
        except Exception:
            continue
    result.sort(key=lambda version: (float(version.created_at or 0.0), str(version.name or "")), reverse=True)
    return [version.model_dump() for version in result]


@router.get("/versions/{version_id}")
def get_version(version_id: str) -> dict[str, Any]:
    data_root = get_data_root()
    version = _load_version(data_root, version_id)
    return version.model_dump()


@router.post("/versions")
def create_version(request: CreateVersionRequest) -> dict[str, Any]:
    """Create a new dataset version."""
    data_root = get_data_root()
    now = time.time()
    version = DatasetVersion(
        version_id=str(uuid.uuid4()),
        name=request.name,
        created_at=now,
        updated_at=now,
        split_assignments=request.split_assignments,
        export_config=request.export_config,
        split_stats=_compute_split_stats(data_root, request.split_assignments, request.export_config),
    )
    _save_version(data_root, version)
    return version.model_dump()


@router.put("/versions/{version_id}")
def update_version(version_id: str, request: UpdateVersionRequest) -> dict[str, Any]:
    """Update an existing dataset version."""
    data_root = get_data_root()
    version = _load_version(data_root, version_id)
    if request.name is not None:
        version.name = request.name
    if request.split_assignments is not None:
        version.split_assignments = request.split_assignments
    if request.export_config is not None:
        version.export_config = request.export_config
    version.updated_at = time.time()
    version.split_stats = _compute_split_stats(data_root, version.split_assignments, version.export_config)
    _save_version(data_root, version)
    return version.model_dump()


@router.delete("/versions/{version_id}")
def delete_version(version_id: str) -> dict[str, bool]:
    """Delete a dataset version."""
    data_root = get_data_root()
    path = _version_path(data_root, version_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")
    path.unlink()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Endpoints — schema fields
# ---------------------------------------------------------------------------


@router.get("/schema-fields")
def get_schema_fields(
    fact_keys: list[str] | None = Query(default=None),
    page_meta_keys: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    """Return available schema fields and a customizable instruction preview."""
    instruction_preview = build_custom_extraction_prompt_template(
        fact_keys=fact_keys,
        page_meta_keys=page_meta_keys,
    )
    return {
        "prompt_fact_keys": list(PROMPT_FACT_KEYS),
        "prompt_page_meta_keys": list(PROMPT_PAGE_META_KEYS),
        "required_prompt_canonical_keys": list(REQUIRED_PROMPT_CANONICAL_KEYS),
        "instruction_preview": instruction_preview,
    }


# ---------------------------------------------------------------------------
# Endpoints — preview
# ---------------------------------------------------------------------------


@router.post("/versions/{version_id}/preview")
def preview_version(version_id: str, request: PreviewRequest) -> dict[str, Any]:
    """Return a sample of rows from each split for preview purposes."""
    data_root = get_data_root()
    version = _load_version(data_root, version_id)
    config = version.export_config

    # Group doc_ids by split, excluding "exclude" entries.
    split_doc_ids: dict[str, list[str]] = {}
    for doc_id, split in version.split_assignments.items():
        if split == "exclude":
            continue
        split_doc_ids.setdefault(split, []).append(doc_id)

    instruction = build_custom_extraction_prompt_template(
        fact_keys=config.selected_fact_keys,
        page_meta_keys=config.selected_page_meta_keys,
        include_bbox=config.include_bbox,
    )

    rows: list[dict[str, Any]] = []
    for split, doc_ids in split_doc_ids.items():
        sampled_doc_ids = doc_ids[: request.n_per_split]
        for doc_id in sampled_doc_ids:
            ann_path = data_root / "annotations" / f"{doc_id}.json"
            if not ann_path.is_file():
                continue
            try:
                from ..schema_io import load_any_schema

                payload = load_any_schema(json.loads(ann_path.read_text("utf-8")))
            except Exception:
                continue
            pages = payload.get("pages") or []
            if not isinstance(pages, list):
                continue
            for page in pages:
                if not isinstance(page, dict):
                    continue
                meta = page.get("meta") or {}
                status = str(meta.get("annotation_status") or "").lower()
                if config.approved_pages_only and status != "approved":
                    continue
                image_name = str(page.get("image") or "").strip()
                if not image_name:
                    continue
                filtered_meta = {k: meta.get(k) for k in config.selected_page_meta_keys}
                filtered_page = {"meta": filtered_meta, "facts": page.get("facts") or []}
                text_preview = json.dumps(filtered_page, ensure_ascii=False)[:500]
                rows.append(
                    {
                        "doc_id": doc_id,
                        "page": image_name,
                        "split": split,
                        "instruction": instruction,
                        "text_preview": text_preview,
                        "image_url": f"/api/images/{doc_id}/pages/{image_name}",
                    }
                )

    return {"rows": rows}


# ---------------------------------------------------------------------------
# Endpoints — push (SSE)
# ---------------------------------------------------------------------------


@router.post("/versions/{version_id}/push")
def push_version(version_id: str):
    """Build and push a dataset version to HuggingFace (SSE stream)."""
    data_root = get_data_root()
    version = _load_version(data_root, version_id)
    config = version.export_config

    def generate() -> Iterator[str]:
        try:
            yield sse_event({"type": "log", "message": "Building dataset rows..."}, event="message")

            # Group doc_ids by split.
            split_doc_ids: dict[str, list[str]] = {}
            for doc_id, split in version.split_assignments.items():
                if split == "exclude":
                    continue
                split_doc_ids.setdefault(split, []).append(doc_id)

            all_rows: list[dict[str, Any]] = []
            for split, doc_ids in split_doc_ids.items():
                rows = _build_split_rows(data_root, doc_ids, split, config)
                all_rows.extend(rows)
                yield sse_event({"type": "log", "message": f"Built {len(rows)} rows for {split}"}, event="message")

            yield sse_event(
                {"type": "log", "message": f"Pushing {len(all_rows)} total rows to HuggingFace..."},
                event="message",
            )

            # Apply image scaling + bbox scaling + compaction via push_dataset_hub internals.
            repos = _push_rows_to_hf(all_rows, config, data_root)
            pushed_at = time.time()
            version.push_status = "pushed"
            version.last_pushed_at = pushed_at
            version.pushed_repos = {str(key): str(value) for key, value in repos.items()}
            version.updated_at = pushed_at
            version.split_stats = _compute_split_stats(data_root, version.split_assignments, version.export_config)
            _save_version(data_root, version)

            yield sse_event({"type": "done", "repos": repos}, event="message")
        except Exception as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="message")

    return sse_response(generate())


def _push_rows_to_hf(
    rows: list[dict[str, Any]],
    config: ExportConfig,
    data_root: Path,
) -> dict[str, Any]:
    """Build a HuggingFace DatasetDict from rows and push it."""
    import os

    from PIL import Image as PILImage

    try:
        from datasets import Dataset, DatasetDict  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: datasets. Install with `pip install datasets`.") from exc

    hf_token = os.environ.get("FINETREE_HF_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "No HuggingFace token found. Set FINETREE_HF_TOKEN, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN."
        )

    repo_id = config.hf_repo.strip() if config.hf_repo else None
    if not repo_id:
        raise RuntimeError("hf_repo must be set in the export config to push to HuggingFace.")

    # Apply image scaling + compact text, build HF feature rows.
    from ..finetune.push_dataset_hub import (  # type: ignore
        _compact_token_text_payload,
        _copy_or_resize_image,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        hf_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for idx, row in enumerate(rows):
            split = row["split"]
            src_image = Path(row["image"])
            dst_image = tmp_path / f"{idx:06d}_{src_image.name}"

            if config.image_scaling and src_image.is_file():
                resize_info = _copy_or_resize_image(
                    src_image,
                    dst_image,
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                )
            elif src_image.is_file():
                shutil.copy2(src_image, dst_image)
                with PILImage.open(src_image) as img:
                    resize_info = {"orig_w": img.width, "orig_h": img.height, "new_w": img.width, "new_h": img.height}
            else:
                continue

            text = row["text"]
            if config.compact_mode in ("compact", "aggressive"):
                remap_keys = config.compact_mode == "aggressive"
                text = _compact_token_text_payload(text, remap_keys=remap_keys)

            hf_row = {
                "image": str(dst_image),
                "system": row["system"],
                "instruction": row["instruction"],
                "text": text,
            }
            hf_rows_by_split.setdefault(split, []).append(hf_row)

        # Build DatasetDict.
        split_datasets: dict[str, Dataset] = {}
        for split, split_rows in hf_rows_by_split.items():
            hf_split_name = "validation" if split == "val" else split
            split_datasets[hf_split_name] = Dataset.from_list(split_rows)
        dataset_dict = DatasetDict(split_datasets)

        # Push.
        repos: dict[str, str] = {}
        if config.push_mode == "separate":
            from ..finetune.push_dataset_hub import push_train_validation_separately  # type: ignore  # noqa: PLC0415

            repos = push_train_validation_separately(
                dataset_dict,
                token=hf_token,
                base_repo_id=repo_id,
            )
        else:
            from ..finetune.push_dataset_hub import push_to_hf  # type: ignore

            pushed_id = push_to_hf(dataset_dict, token=hf_token, repo_id=repo_id)
            repos["repo"] = pushed_id

    return repos


# ---------------------------------------------------------------------------
# Endpoints — token stats (SSE)
# ---------------------------------------------------------------------------


@router.post("/versions/{version_id}/token-stats")
def token_stats_version(version_id: str, request: TokenStatsRequest):
    """Compute token statistics for a dataset version (SSE stream)."""
    data_root = get_data_root()
    version = _load_version(data_root, version_id)
    config = version.export_config

    def generate() -> Iterator[str]:
        try:
            # Group doc_ids by split, excluding "exclude" entries.
            split_doc_ids: dict[str, list[str]] = {}
            for doc_id, split in version.split_assignments.items():
                if split == "exclude":
                    continue
                split_doc_ids.setdefault(split, []).append(doc_id)

            yield sse_event({"type": "log", "message": "Building rows for token counting..."}, event="message")

            # Write rows to temp JSONL files and run token stats.
            all_rows: list[dict[str, Any]] = []
            for split, doc_ids in split_doc_ids.items():
                rows = _build_split_rows(data_root, doc_ids, split, config)
                all_rows.extend(rows)

            total = len(all_rows)
            yield sse_event(
                {"type": "log", "message": f"Tokenizing {total} samples with {request.tokenizer}..."},
                event="message",
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_jsonl = Path(tmp_dir) / "rows.jsonl"
                tmp_jsonl.write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows),
                    encoding="utf-8",
                )

                # Use a threading event to pass progress updates back to the generator.
                progress_queue: list[tuple[int, int]] = []
                lock = threading.Lock()

                def progress_cb(current: int, total_count: int) -> None:
                    with lock:
                        progress_queue.append((current, total_count))

                result_holder: list[dict[str, Any]] = []
                error_holder: list[Exception] = []

                def run_stats() -> None:
                    try:
                        from ..finetune.token_stats import compute_jsonl_token_stats

                        stats = compute_jsonl_token_stats(
                            [tmp_jsonl],
                            tokenizer_name=request.tokenizer,
                            progress_callback=progress_cb,
                        )
                        result_holder.append(stats)
                    except Exception as exc:
                        error_holder.append(exc)

                t = threading.Thread(target=run_stats, daemon=True)
                t.start()

                while t.is_alive():
                    t.join(timeout=0.5)
                    with lock:
                        pending = list(progress_queue)
                        progress_queue.clear()
                    for current, total_count in pending:
                        yield sse_event(
                            {"type": "log", "message": f"Tokenized {current}/{total_count} samples"},
                            event="message",
                        )

                # Drain any remaining progress events.
                with lock:
                    pending = list(progress_queue)
                for current, total_count in pending:
                    yield sse_event(
                        {"type": "log", "message": f"Tokenized {current}/{total_count} samples"},
                        event="message",
                    )

            if error_holder:
                raise error_holder[0]

            stats_result = result_holder[0] if result_holder else {}
            yield sse_event({"type": "done", "result": stats_result}, event="message")
        except Exception as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="message")

    return sse_response(generate())


# ---------------------------------------------------------------------------
# Stateless endpoints (no version_id required — used by the UI for live preview,
# push, and token stats without needing to save a version first)
# ---------------------------------------------------------------------------


@router.post("/preview")
def preview_stateless(request: StatelessPreviewRequest) -> dict[str, Any]:
    """Return sample rows for an inline split+config without a saved version."""
    data_root = get_data_root()
    config = request.export_config
    target_split = request.split

    instruction = build_custom_extraction_prompt_template(
        fact_keys=config.selected_fact_keys,
        page_meta_keys=config.selected_page_meta_keys,
        include_bbox=config.include_bbox,
    )

    doc_ids = [
        doc_id
        for doc_id, split in request.assignments.items()
        if split == target_split
    ][: request.n_per_split]

    rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        ann_path = data_root / "annotations" / f"{doc_id}.json"
        if not ann_path.is_file():
            continue
        try:
            from ..schema_io import load_any_schema
            payload = load_any_schema(json.loads(ann_path.read_text("utf-8")))
        except Exception:
            continue
        pages = payload.get("pages") or []
        for page in pages:
            if not isinstance(page, dict):
                continue
            meta = page.get("meta") or {}
            status = str(meta.get("annotation_status") or "").lower()
            if config.approved_pages_only and status != "approved":
                continue
            image_name = str(page.get("image") or "").strip()
            if not image_name:
                continue
            filtered_meta = {k: meta.get(k) for k in config.selected_page_meta_keys}
            filtered_page = {"meta": filtered_meta, "facts": page.get("facts") or []}
            text_preview = json.dumps(filtered_page, ensure_ascii=False)[:500]
            rows.append({
                "doc_id": doc_id,
                "page": image_name,
                "split": target_split,
                "instruction": instruction,
                "text_preview": text_preview,
                "image_url": f"/api/images/{doc_id}/pages/{image_name}",
            })
            break  # one page per doc for preview

    return {"rows": rows}


@router.post("/push")
def push_stateless(request: StatelessPushRequest):
    """Build and push a dataset to HuggingFace from inline assignments (SSE stream)."""
    data_root = get_data_root()
    config = request.export_config

    def generate() -> Iterator[str]:
        try:
            yield sse_event({"type": "log", "message": "Building dataset rows..."}, event="message")
            split_doc_ids: dict[str, list[str]] = {}
            for doc_id, split in request.assignments.items():
                if split == "exclude":
                    continue
                split_doc_ids.setdefault(split, []).append(doc_id)

            all_rows: list[dict[str, Any]] = []
            for split, doc_ids in split_doc_ids.items():
                rows = _build_split_rows(data_root, doc_ids, split, config)
                all_rows.extend(rows)
                yield sse_event({"type": "log", "message": f"Built {len(rows)} rows for {split}"}, event="message")

            yield sse_event(
                {"type": "log", "message": f"Pushing {len(all_rows)} total rows to HuggingFace..."},
                event="message",
            )
            repos = _push_rows_to_hf(all_rows, config, data_root)
            yield sse_event({"type": "done", "repos": repos}, event="message")
        except Exception as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="message")

    return sse_response(generate())


@router.post("/token-stats")
def token_stats_stateless(request: StatelessTokenStatsRequest):
    """Compute token statistics from inline assignments (SSE stream)."""
    data_root = get_data_root()
    config = request.export_config

    def generate() -> Iterator[str]:
        try:
            split_doc_ids: dict[str, list[str]] = {}
            for doc_id, split in request.assignments.items():
                if split == "exclude":
                    continue
                split_doc_ids.setdefault(split, []).append(doc_id)

            yield sse_event({"type": "log", "message": "Building rows for token counting..."}, event="message")
            all_rows: list[dict[str, Any]] = []
            for split, doc_ids in split_doc_ids.items():
                rows = _build_split_rows(data_root, doc_ids, split, config)
                all_rows.extend(rows)

            total = len(all_rows)
            yield sse_event(
                {"type": "log", "message": f"Tokenizing {total} samples with {request.tokenizer}..."},
                event="message",
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_jsonl = Path(tmp_dir) / "rows.jsonl"
                tmp_jsonl.write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows),
                    encoding="utf-8",
                )

                progress_queue: list[tuple[int, int]] = []
                lock = threading.Lock()

                def progress_cb(current: int, total_count: int) -> None:
                    with lock:
                        progress_queue.append((current, total_count))

                result_holder: list[dict[str, Any]] = []
                error_holder: list[Exception] = []

                def run_stats() -> None:
                    try:
                        from ..finetune.token_stats import compute_jsonl_token_stats
                        stats = compute_jsonl_token_stats(
                            [tmp_jsonl],
                            tokenizer_name=request.tokenizer,
                            progress_callback=progress_cb,
                        )
                        result_holder.append(stats)
                    except Exception as exc:
                        error_holder.append(exc)

                t = threading.Thread(target=run_stats, daemon=True)
                t.start()
                while t.is_alive():
                    t.join(timeout=0.5)
                    with lock:
                        pending = list(progress_queue)
                        progress_queue.clear()
                    for current, total_count in pending:
                        yield sse_event(
                            {"type": "log", "message": f"Tokenized {current}/{total_count} samples"},
                            event="message",
                        )
                with lock:
                    pending = list(progress_queue)
                for current, total_count in pending:
                    yield sse_event(
                        {"type": "log", "message": f"Tokenized {current}/{total_count} samples"},
                        event="message",
                    )

            if error_holder:
                raise error_holder[0]

            stats_result = result_holder[0] if result_holder else {}
            yield sse_event({"type": "done", "result": stats_result}, event="message")
        except Exception as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="message")

    return sse_response(generate())
