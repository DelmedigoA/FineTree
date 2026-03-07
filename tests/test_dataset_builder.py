from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.finetune.config import FinetuneConfig
from finetree_annotator.finetune.dataset_builder import build_unsloth_chat_datasets


def _write_annotation(annotation_path: Path, image_dir: Path, image_name: str) -> None:
    payload = {
        "images_dir": str(image_dir),
        "metadata": {
            "language": None,
            "reading_direction": None,
            "company_name": None,
            "company_id": None,
            "report_year": None,
            "entity_type": None,
        },
        "pages": [
            {
                "image": image_name,
                "meta": {
                    "entity_name": "ACME",
                    "page_num": "1",
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                },
                "facts": [
                    {
                        "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                        "value": "10",
                        "note_ref": None,
                        "comment_ref": None,
                        "date": None,
                        "period_type": None,
                        "period_start": None,
                        "period_end": None,
                        "path": ["assets"],
                        "path_source": None,
                        "note_flag": False,
                        "note_name": None,
                        "note_num": None,
                        "currency": "ILS",
                        "scale": 1,
                        "value_type": "amount",
                    }
                ],
            }
        ],
    }
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_dataset_builder_writes_unsloth_chat_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    ann_dir = data_dir / "annotations"
    img_dir = data_dir / "pdf_images" / "doc1"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    image_name = "page_0001.png"
    (img_dir / image_name).write_bytes(b"fake")
    _write_annotation(ann_dir / "doc1.json", image_dir=img_dir, image_name=image_name)

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
                "output_train_jsonl": "data/finetune/train.jsonl",
                "output_val_jsonl": "data/finetune/val.jsonl",
                "val_ratio": 0.0,
            },
            "prompt": {
                "use_custom_prompt": False,
                "fallback_template": "Prompt for {{IMAGE_NAME}}",
            },
        }
    )

    stats = build_unsloth_chat_datasets(cfg)
    assert stats.samples_written_train == 1
    assert stats.samples_written_val == 0

    lines = (tmp_path / "data/finetune/train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    sample = json.loads(lines[0])
    assert sample["messages"][0]["content"][0]["type"] == "image"
    assert sample["messages"][1]["content"][0]["type"] == "text"
    assert sample["metadata"]["reading_direction"] in {"rtl", "ltr"}
    assert "reading_direction_source" in sample["metadata"]
    assert "reading_direction_uncertain" in sample["metadata"]
    out_obj = json.loads(sample["messages"][1]["content"][0]["text"])
    assert out_obj["metadata"]["entity_type"] is None
    assert out_obj["pages"][0]["facts"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_dataset_builder_can_drop_bbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    ann_dir = data_dir / "annotations"
    img_dir = data_dir / "pdf_images" / "doc2"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    image_name = "page_0001.png"
    (img_dir / image_name).write_bytes(b"fake")
    _write_annotation(ann_dir / "doc2.json", image_dir=img_dir, image_name=image_name)

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
                "output_train_jsonl": "data/finetune/train.jsonl",
                "output_val_jsonl": "data/finetune/val.jsonl",
                "val_ratio": 0.0,
                "bbox_policy": "drop_all",
            },
            "prompt": {
                "use_custom_prompt": False,
            },
        }
    )

    build_unsloth_chat_datasets(cfg)
    line = (tmp_path / "data/finetune/train.jsonl").read_text(encoding="utf-8").strip()
    sample = json.loads(line)
    out_obj = json.loads(sample["messages"][1]["content"][0]["text"])
    assert "bbox" not in out_obj["pages"][0]["facts"][0]


def test_dataset_builder_doc_split_map_ensures_non_empty_val_for_multi_doc(monkeypatch) -> None:
    from finetree_annotator.finetune import dataset_builder as mod

    docs = [Path("a.json"), Path("b.json"), Path("c.json")]
    monkeypatch.setattr(mod, "_doc_in_val_split", lambda _doc, _ratio: False)
    split = mod._doc_split_map(docs, val_ratio=0.1)
    assert any(split.values())
    assert not all(split.values())


def test_dataset_builder_doc_split_map_can_force_explicit_val_docs() -> None:
    from finetree_annotator.finetune import dataset_builder as mod

    docs = [Path("pdf_2.json"), Path("pdf_3.json"), Path("pdf_4.json"), Path("test.json")]
    split = mod._doc_split_map(docs, val_ratio=0.1, forced_val_doc_ids={"pdf_4"})
    assert split == {
        "pdf_2": False,
        "pdf_3": False,
        "pdf_4": True,
        "test": False,
    }


def test_dataset_builder_doc_split_map_can_force_empty_explicit_validation() -> None:
    from finetree_annotator.finetune import dataset_builder as mod

    docs = [Path("pdf_2.json"), Path("pdf_3.json")]
    split = mod._doc_split_map(
        docs,
        val_ratio=0.4,
        forced_val_doc_ids=set(),
        force_explicit_val_doc_ids=True,
    )
    assert split == {
        "pdf_2": False,
        "pdf_3": False,
    }


def test_dataset_builder_reorders_hebrew_row_right_to_left(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    ann_dir = data_dir / "annotations"
    img_dir = data_dir / "pdf_images" / "doc_he"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    image_name = "page_0001.png"
    (img_dir / image_name).write_bytes(b"fake")
    payload = {
        "images_dir": str(img_dir),
        "document_meta": {"language": "he"},
        "pages": [
            {
                "image": image_name,
                "meta": {
                    "entity_name": "חברה",
                    "page_num": "1",
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                },
                "facts": [
                    {
                        "bbox": {"x": 10, "y": 10, "w": 10, "h": 10},
                        "value": "left",
                        "note_ref": None,
                        "comment_ref": None,
                        "note_flag": False,
                        "note_name": None,
                        "note_num": None,
                        "date": None,
                        "period_type": None,
                        "period_start": None,
                        "period_end": None,
                        "path": [],
                        "path_source": None,
                    },
                    {
                        "bbox": {"x": 100, "y": 10, "w": 10, "h": 10},
                        "value": "right",
                        "note_ref": None,
                        "comment_ref": None,
                        "note_flag": False,
                        "note_name": None,
                        "note_num": None,
                        "date": None,
                        "period_type": None,
                        "period_start": None,
                        "period_end": None,
                        "path": [],
                        "path_source": None,
                    },
                ],
            }
        ],
    }
    (ann_dir / "doc_he.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
                "output_train_jsonl": "data/finetune/train.jsonl",
                "output_val_jsonl": "data/finetune/val.jsonl",
                "val_ratio": 0.0,
            },
            "prompt": {"use_custom_prompt": False},
        }
    )
    build_unsloth_chat_datasets(cfg)

    line = (tmp_path / "data/finetune/train.jsonl").read_text(encoding="utf-8").strip()
    sample = json.loads(line)
    out_obj = json.loads(sample["messages"][1]["content"][0]["text"])
    assert [fact["value"] for fact in out_obj["pages"][0]["facts"]] == ["right", "left"]


def test_dataset_builder_can_limit_to_reviewed_docs_and_explicit_validation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    ann_dir = data_dir / "annotations"
    img_root = data_dir / "pdf_images"
    ann_dir.mkdir(parents=True)
    img_root.mkdir(parents=True)

    for doc_id in ("doc_a", "doc_b"):
        img_dir = img_root / doc_id
        img_dir.mkdir(parents=True)
        image_name = "page_0001.png"
        (img_dir / image_name).write_bytes(b"fake")
        _write_annotation(ann_dir / f"{doc_id}.json", image_dir=img_dir, image_name=image_name)

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
                "output_train_jsonl": "data/finetune/train.jsonl",
                "output_val_jsonl": "data/finetune/val.jsonl",
                "val_ratio": 0.5,
            },
            "prompt": {"use_custom_prompt": False},
        }
    )

    stats = build_unsloth_chat_datasets(
        cfg,
        include_doc_ids={"doc_a"},
        forced_val_doc_ids=set(),
        force_explicit_val_doc_ids=True,
    )

    assert stats.annotation_files == 1
    assert stats.samples_written_train == 1
    assert stats.samples_written_val == 0
    train_lines = (tmp_path / "data/finetune/train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 1
    sample = json.loads(train_lines[0])
    assert sample["metadata"]["document_id"] == "doc_a"
    assert (tmp_path / "data/finetune/val.jsonl").read_text(encoding="utf-8") == ""
