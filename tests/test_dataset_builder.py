from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.finetune.config import FinetuneConfig
from finetree_annotator.finetune.dataset_builder import build_unsloth_chat_datasets


def _write_annotation(annotation_path: Path, image_dir: Path, image_name: str) -> None:
    payload = {
        "images_dir": str(image_dir),
        "pages": [
            {
                "image": image_name,
                "meta": {"entity_name": "ACME", "page_num": "1", "type": "other", "title": None},
                "facts": [
                    {
                        "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                        "value": "10",
                        "refference": "",
                        "date": None,
                        "path": ["assets"],
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
    out_obj = json.loads(sample["messages"][1]["content"][0]["text"])
    assert out_obj["facts"][0]["bbox"]["x"] == 1


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
    assert "bbox" not in out_obj["facts"][0]


def test_dataset_builder_doc_split_map_ensures_non_empty_val_for_multi_doc(monkeypatch) -> None:
    from finetree_annotator.finetune import dataset_builder as mod

    docs = [Path("a.json"), Path("b.json"), Path("c.json")]
    monkeypatch.setattr(mod, "_doc_in_val_split", lambda _doc, _ratio: False)
    split = mod._doc_split_map(docs, val_ratio=0.1)
    assert any(split.values())
    assert not all(split.values())
