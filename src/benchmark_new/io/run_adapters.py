from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workspace_paths import DEFAULT_DATA_ROOT, resolve_annotations_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _annotation_doc_exists(doc_id: str, data_root: Path) -> bool:
    try:
        return resolve_annotations_path(doc_id, data_root=data_root).is_file()
    except Exception:
        return False


def _resolve_doc_id(raw_doc_id: str, *, data_root: Path) -> str:
    candidate = str(raw_doc_id or "").strip()
    if candidate and _annotation_doc_exists(candidate, data_root):
        return candidate
    relaxed = candidate.replace(" ", "_")
    if relaxed and _annotation_doc_exists(relaxed, data_root):
        return relaxed
    return candidate or relaxed


def _normalize_page_from_row(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("parsed_page"), dict):
        return dict(row["parsed_page"])
    if isinstance(row.get("parsed_json"), dict):
        parsed_json = row["parsed_json"]
        if "meta" in parsed_json or "facts" in parsed_json:
            return dict(parsed_json)
        pages = parsed_json.get("pages") if isinstance(parsed_json.get("pages"), list) else []
        if pages and isinstance(pages[0], dict):
            return dict(pages[0])
    assistant_text = row.get("assistant_text")
    if isinstance(assistant_text, str) and assistant_text.strip():
        try:
            decoded = json.loads(assistant_text)
        except Exception as exc:
            raise RuntimeError(f"assistant_text_json_error: {exc}") from exc
        if isinstance(decoded, dict) and ("meta" in decoded or "facts" in decoded):
            return decoded
        pages = decoded.get("pages") if isinstance(decoded, dict) and isinstance(decoded.get("pages"), list) else []
        if pages and isinstance(pages[0], dict):
            return dict(pages[0])
    raise RuntimeError("Prediction row does not contain a usable page payload.")


def _group_rows(rows: list[dict[str, Any]], *, data_root: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_doc_id = str(row.get("doc_id") or "").strip()
        doc_id = _resolve_doc_id(raw_doc_id, data_root=data_root) if raw_doc_id else ""
        if not doc_id:
            continue
        page_index = int(row.get("page_index") or row.get("page") or 0)
        if page_index < 1:
            image_name = str(row.get("page_name") or row.get("image_name") or "")
            if image_name.startswith("page_") and image_name.endswith(".png"):
                page_index = int(image_name[5:9])
        row_error = str(row.get("error") or "").strip() or None
        try:
            page_payload = _normalize_page_from_row(row)
        except Exception as exc:
            page_payload = None
            row_error = row_error or str(exc)
        if isinstance(page_payload, dict):
            page_payload.setdefault("image", str(row.get("page_name") or row.get("image_name") or page_payload.get("image") or f"page_{page_index:04d}.png"))
        documents.setdefault(doc_id, {"doc_id": doc_id, "pages": []})["pages"].append(
            {
                "page_index": page_index,
                "page_name": (page_payload or {}).get("image") or str(row.get("page_name") or row.get("image_name") or f"page_{page_index:04d}.png"),
                "assistant_text": row.get("assistant_text"),
                "parsed_page": page_payload,
                "error": row_error,
                "received_tokens": int(row.get("received_tokens") or 0),
            }
        )
    for document in documents.values():
        document["pages"].sort(key=lambda item: int(item["page_index"]))
    return documents


def _load_native_run(run_dir: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    docs_dir = run_dir / "documents"
    for doc_dir in sorted(path for path in docs_dir.iterdir() if path.is_dir()):
        pages_dir = doc_dir / "pages"
        if not pages_dir.is_dir():
            continue
        pages: list[dict[str, Any]] = []
        for page_file in sorted(pages_dir.glob("page_*.json")):
            row = _load_json(page_file)
            page_index = int(row.get("page_index") or 0)
            pages.append(
                {
                    "page_index": page_index,
                    "page_name": str(row.get("page_name") or row.get("image_name") or f"page_{page_index:04d}.png"),
                    "assistant_text": row.get("assistant_text"),
                    "parsed_page": row.get("parsed_page"),
                    "error": row.get("error"),
                    "received_tokens": int(row.get("received_tokens") or 0),
                }
            )
        documents[doc_dir.name] = {"doc_id": doc_dir.name, "pages": pages}
    manifest = _load_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
    return {"kind": "native", "manifest": manifest, "documents": documents}


def _load_gui_batch_run(run_dir: Path, *, data_root: Path) -> dict[str, Any]:
    requests_path = run_dir / "requests.jsonl"
    rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = _load_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
    return {"kind": "gui_batch_run", "manifest": manifest, "documents": _group_rows(rows, data_root=data_root)}


def _load_vllm_leaf_run(run_dir: Path, *, data_root: Path) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    results_path = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc_id = _resolve_doc_id(Path(str(manifest.get("pdf") or "")).stem, data_root=data_root)
    for row in rows:
        row["doc_id"] = doc_id
    return {"kind": "vllm_results", "manifest": manifest, "documents": _group_rows(rows, data_root=data_root)}


def _load_vllm_multi_run(run_dir: Path, *, data_root: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    manifest = _load_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
    for child in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        if not (child / "manifest.json").is_file() or not (child / "results.jsonl").is_file():
            continue
        loaded = _load_vllm_leaf_run(child, data_root=data_root)
        documents.update(loaded["documents"])
    return {"kind": "vllm_multi_results", "manifest": manifest, "documents": documents}


def load_predictions_from_run_dir(run_dir: Path, *, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if (run_dir / "documents").is_dir():
        return _load_native_run(run_dir)
    if (run_dir / "requests.jsonl").is_file():
        return _load_gui_batch_run(run_dir, data_root=Path(data_root))
    if (run_dir / "results.jsonl").is_file() and (run_dir / "manifest.json").is_file():
        return _load_vllm_leaf_run(run_dir, data_root=Path(data_root))
    if (run_dir / "summary.json").is_file():
        return _load_vllm_multi_run(run_dir, data_root=Path(data_root))
    raise RuntimeError(f"Unsupported run directory format: {run_dir}")
