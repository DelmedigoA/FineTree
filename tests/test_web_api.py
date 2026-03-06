from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from finetree_annotator.annotation_core import BoxRecord, PageState, build_annotations_payload, serialize_annotations_json
from finetree_annotator.web.api import create_app
from finetree_annotator.workspace import annotations_root, pdf_images_root, raw_pdfs_dir


def _write_png(path: Path, *, width: int = 200, height: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color="#ffffff").save(path, format="PNG")


def _seed_workspace(data_root: Path) -> Path:
    doc_id = "doc_1"
    images_dir = pdf_images_root(data_root) / doc_id
    annotations_path = annotations_root(data_root) / f"{doc_id}.json"
    raw_pdfs_dir(data_root).mkdir(parents=True, exist_ok=True)
    (raw_pdfs_dir(data_root) / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4")
    _write_png(images_dir / "page_0001.png", width=200, height=120)
    _write_png(images_dir / "page_0002.png", width=200, height=120)

    payload = build_annotations_payload(
        images_dir,
        [images_dir / "page_0001.png", images_dir / "page_0002.png"],
        {
            "page_0001.png": PageState(
                meta={"entity_name": "Acme", "page_num": "1", "type": "other", "title": "Page 1"},
                facts=[
                    BoxRecord(
                        bbox={"x": 120, "y": 10, "w": 20, "h": 12},
                        fact={"value": "200", "path": ["Revenue"], "date": "2024-12-31"},
                    ),
                    BoxRecord(
                        bbox={"x": 20, "y": 10, "w": 20, "h": 12},
                        fact={"value": "100", "path": ["Revenue"], "date": "2023-12-31"},
                    ),
                ],
            ),
            "page_0002.png": PageState(
                meta={"entity_name": None, "page_num": "2", "type": "notes", "title": "Notes"},
                facts=[],
            ),
        },
        document_meta={"language": "en", "reading_direction": None, "company_name": "Acme", "report_year": 2024},
    )
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(serialize_annotations_json(payload), encoding="utf-8")
    return annotations_path


def test_workspace_list_and_document_detail(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _seed_workspace(data_root)
    app = create_app(data_root=data_root)
    client = TestClient(app)

    docs = client.get("/api/workspace/documents")
    assert docs.status_code == 200
    payload = docs.json()
    assert payload[0]["doc_id"] == "doc_1"
    assert payload[0]["page_count"] == 2

    detail = client.get("/api/documents/doc_1")
    assert detail.status_code == 200
    body = detail.json()
    assert body["document_meta"]["language"] == "en"
    assert body["pages"][0]["facts"][0]["value"] == "100"
    assert body["pages"][0]["facts"][1]["value"] == "200"
    assert body["pages"][1]["issue_summary"]["warning_count"] == 0


def test_save_document_and_validate(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_path = _seed_workspace(data_root)
    app = create_app(data_root=data_root)
    client = TestClient(app)

    detail = client.get("/api/documents/doc_1").json()
    detail["pages"][0]["facts"][0]["value"] = "900"
    validate_resp = client.post(
        "/api/documents/doc_1/validate",
        json={"document_meta": detail["document_meta"], "pages": [{"image": p["image"], "meta": p["meta"], "facts": p["facts"]} for p in detail["pages"]]},
    )
    assert validate_resp.status_code == 200
    assert validate_resp.json()["issue_summary"]["pages_with_warnings"] >= 0

    save_resp = client.put(
        "/api/documents/doc_1",
        json={"document_meta": detail["document_meta"], "pages": [{"image": p["image"], "meta": p["meta"], "facts": p["facts"]} for p in detail["pages"]]},
    )
    assert save_resp.status_code == 200
    saved = json.loads(annotations_path.read_text(encoding="utf-8"))
    assert saved["pages"][0]["facts"][0]["value"] == "900"
    assert save_resp.json()["document"]["pages"][0]["facts"][0]["value"] == "900"


def test_import_json_converts_normalized_bbox_and_apply_entity(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_path = _seed_workspace(data_root)
    app = create_app(data_root=data_root)
    client = TestClient(app)

    import_resp = client.post(
        "/api/documents/doc_1/import-json",
        json={
            "payload": {
                "pages": [
                    {
                        "image": "page_0002.png",
                        "meta": {"entity_name": None, "page_num": "2", "type": "notes", "title": "Notes"},
                        "facts": [
                            {
                                "bbox": {"x": 100, "y": 100, "w": 200, "h": 200},
                                "value": "77",
                                "path": ["Liabilities"],
                                "note_reference": None,
                            }
                        ],
                    }
                ]
            },
            "normalized_1000": True,
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    bbox = imported["pages"][1]["facts"][0]["bbox"]
    assert bbox["x"] == 20.0
    assert bbox["w"] == 40.0

    apply_resp = client.post(
        "/api/documents/doc_1/apply-entity",
        json={"entity_name": "Acme LLC", "overwrite_existing": False},
    )
    assert apply_resp.status_code == 200
    applied = apply_resp.json()
    assert applied["pages"][1]["meta"]["entity_name"] == "Acme LLC"

    saved = json.loads(annotations_path.read_text(encoding="utf-8"))
    assert saved["pages"][1]["meta"]["entity_name"] == "Acme LLC"


def test_import_pdf_upload_endpoint_rejects_invalid_base64(tmp_path: Path) -> None:
    app = create_app(data_root=tmp_path / "data")
    client = TestClient(app)

    response = client.post(
        "/api/workspace/import-pdf",
        json={"filename": "doc.pdf", "content_b64": "not-base64", "dpi": 200},
    )
    assert response.status_code == 400
    assert "Invalid base64" in response.json()["detail"]


def test_app_config_and_schema_endpoint(tmp_path: Path) -> None:
    app = create_app(data_root=tmp_path / "data", startup_doc_id="doc_1")
    client = TestClient(app)

    config = client.get("/api/app-config")
    assert config.status_code == 200
    assert config.json()["startup_doc_id"] == "doc_1"

    schema = client.get("/api/schema")
    assert schema.status_code == 200
    assert "page_types" in schema.json()["schema"]
