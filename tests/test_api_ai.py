from __future__ import annotations

import json

from fastapi.testclient import TestClient

from finetree_annotator.api.app import create_app


def _make_client(tmp_path, doc_id: str, page_name: str) -> TestClient:
    images_dir = tmp_path / "pdf_images" / doc_id
    images_dir.mkdir(parents=True)
    (images_dir / page_name).write_bytes(b"fake-image")

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / f"{doc_id}.json").write_text(
        json.dumps({"document_meta": {"reading_direction": "rtl"}}),
        encoding="utf-8",
    )
    return TestClient(create_app(data_root=tmp_path))


def test_align_bboxes_returns_page_indexed_box_records(tmp_path, monkeypatch) -> None:
    doc_id = "doc_1"
    page_name = "page_0001.png"

    monkeypatch.setattr(
        "finetree_annotator.local_doctr.extract_numeric_bbox_facts",
        lambda _image_path: [{"bbox": {"x": 11, "y": 22, "w": 33, "h": 44}, "value": "100"}],
    )
    monkeypatch.setattr(
        "finetree_annotator.qwen_import_matcher.match_qwen_import_payloads",
        lambda **_kwargs: (
            [
                {
                    "page_index": 3,
                    "bbox": {"x": 11, "y": 22, "w": 33, "h": 44},
                    "value": "100",
                    "fact_num": 7,
                    "currency": "ILS",
                }
            ],
            {},
        ),
    )

    client = _make_client(tmp_path, doc_id, page_name)
    response = client.post(
        "/api/ai/align-bboxes",
        json={
            "doc_id": doc_id,
            "page_name": page_name,
            "facts": [
                {
                    "page_index": 3,
                    "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                    "value": "100",
                    "fact_num": 7,
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["aligned_facts"]) == 1
    aligned_fact = payload["aligned_facts"][0]
    assert aligned_fact["page_index"] == 3
    assert aligned_fact["bbox"] == {"x": 11, "y": 22, "w": 33, "h": 44}
    assert aligned_fact["fact"]["value"] == "100"
    assert aligned_fact["fact"]["fact_num"] == 7
    assert aligned_fact["fact"]["currency"] == "ILS"
    assert aligned_fact["fact"]["row_role"] == "detail"
    assert "bbox" not in aligned_fact["fact"]


def test_align_bboxes_subset_returns_only_targeted_page_indices(tmp_path, monkeypatch) -> None:
    doc_id = "doc_1"
    page_name = "page_0001.png"

    monkeypatch.setattr(
        "finetree_annotator.local_doctr.extract_numeric_bbox_facts",
        lambda _image_path: [{"bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "value": "100"}],
    )
    monkeypatch.setattr(
        "finetree_annotator.qwen_import_matcher.match_qwen_import_payloads",
        lambda **_kwargs: (
            [
                {
                    "page_index": 2,
                    "bbox": {"x": 20, "y": 10, "w": 5, "h": 5},
                    "value": "100",
                    "fact_num": 10,
                },
                {
                    "page_index": 7,
                    "bbox": {"x": 70, "y": 35, "w": 8, "h": 8},
                    "value": "200",
                    "fact_num": 20,
                },
            ],
            {},
        ),
    )

    client = _make_client(tmp_path, doc_id, page_name)
    response = client.post(
        "/api/ai/align-bboxes",
        json={
            "doc_id": doc_id,
            "page_name": page_name,
            "facts": [
                {"page_index": 2, "bbox": {"x": 1, "y": 1, "w": 1, "h": 1}, "value": "100", "fact_num": 10},
                {"page_index": 7, "bbox": {"x": 2, "y": 2, "w": 2, "h": 2}, "value": "200", "fact_num": 20},
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [item["page_index"] for item in payload["aligned_facts"]] == [2, 7]
    assert payload["aligned_facts"][0]["bbox"] == {"x": 20, "y": 10, "w": 5, "h": 5}
    assert payload["aligned_facts"][1]["bbox"] == {"x": 70, "y": 35, "w": 8, "h": 8}


def test_align_bboxes_preserves_page_index_when_matcher_reorders_subset(tmp_path, monkeypatch) -> None:
    doc_id = "doc_1"
    page_name = "page_0001.png"

    monkeypatch.setattr(
        "finetree_annotator.local_doctr.extract_numeric_bbox_facts",
        lambda _image_path: [{"bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "value": "100"}],
    )

    def _reordered_matcher(**kwargs):
        imported_payloads = list(kwargs["imported_payloads"])
        reordered = sorted(imported_payloads, key=lambda payload: int(payload["fact_num"]))
        for offset, payload in enumerate(reordered):
            payload["bbox"] = {"x": 100 + offset, "y": 200 + offset, "w": 30, "h": 40}
        return reordered, {}

    monkeypatch.setattr(
        "finetree_annotator.qwen_import_matcher.match_qwen_import_payloads",
        _reordered_matcher,
    )

    client = _make_client(tmp_path, doc_id, page_name)
    response = client.post(
        "/api/ai/align-bboxes",
        json={
            "doc_id": doc_id,
            "page_name": page_name,
            "facts": [
                {"page_index": 5, "bbox": {"x": 9, "y": 9, "w": 9, "h": 9}, "value": "200", "fact_num": 2},
                {"page_index": 1, "bbox": {"x": 4, "y": 4, "w": 4, "h": 4}, "value": "100", "fact_num": 1},
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [(item["page_index"], item["fact"]["value"]) for item in payload["aligned_facts"]] == [
        (1, "100"),
        (5, "200"),
    ]
    assert payload["aligned_facts"][0]["bbox"] == {"x": 100, "y": 200, "w": 30, "h": 40}
    assert payload["aligned_facts"][1]["bbox"] == {"x": 101, "y": 201, "w": 30, "h": 40}
