from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from finetree_annotator.finetune import push_dataset_hub_no_bbox as no_bbox_mod


def _install_fake_qwen_vl_utils(monkeypatch, scale: float = 0.5) -> None:
    vision_mod = types.ModuleType("qwen_vl_utils.vision_process")

    def _smart_resize(height: int, width: int, **kwargs):
        _ = kwargs
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    vision_mod.smart_resize = _smart_resize
    root_mod = types.ModuleType("qwen_vl_utils")
    root_mod.vision_process = vision_mod
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", root_mod)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_mod)


def _empty_dataset() -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    empty = {"image": [], "instruction": [], "text": []}
    return DatasetDict(
        {
            "train": Dataset.from_dict(empty, features=features),
            "validation": Dataset.from_dict(empty, features=features),
        }
    )


def test_export_for_hf_no_bbox_strips_bbox_and_sanitizes_source_instruction(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    target_payload = {
        "meta": {"type": "other"},
        "facts": [
            {
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "value": "10",
                "refference": "",
                "path": ["a"],
            }
        ],
    }
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {
                        "type": "text",
                        "text": (
                            "Return JSON.\n"
                            '"bbox": { "x": <number> }\n'
                            "bbox must be tightly aligned.\n"
                            "Keep facts in order."
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target_payload)}]},
        ],
        "metadata": {"document_id": "doc1"},
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export_no_bbox"
    train_rows, val_rows = no_bbox_mod.export_for_hf_no_bbox(root, export_dir, instruction_mode="source")

    assert train_rows == 1
    assert val_rows == 0
    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert out["image"] == "images/doc1/page_0001.png"
    assert "bbox" not in out["instruction"].lower()
    assert "Keep facts in order." in out["instruction"]

    output_payload = json.loads(out["text"])
    assert "bbox" not in output_payload["facts"][0]


def test_export_for_hf_no_bbox_minimal_instruction_is_fixed_line(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Any long instruction with bbox mentions and formatting."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": '{"meta":{"type":"other"},"facts":[]}'}]},
        ],
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export_no_bbox"
    train_rows, _ = no_bbox_mod.export_for_hf_no_bbox(root, export_dir, instruction_mode="minimal")
    assert train_rows == 1

    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert out["instruction"] == no_bbox_mod._MINIMAL_INSTRUCTION


def test_export_for_hf_no_bbox_compact_tokens_shortens_keys_and_keeps_nulls(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    payload = {
        "meta": {"entity_name": "X", "page_num": "8", "title": "T"},
        "facts": [
            {
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "value": "9,876",
                "date": "30.09.2021",
                "currency": "ILS",
                "scale": 1000,
                "value_type": "amount",
                "refference": None,
                "note": None,
            }
        ],
    }
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Prompt"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export_no_bbox"
    train_rows, _ = no_bbox_mod.export_for_hf_no_bbox(root, export_dir, instruction_mode="source", compact_tokens=True)
    assert train_rows == 1

    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    compact_text = out["text"]
    assert ": " not in compact_text
    assert ", " not in compact_text

    compact_payload = json.loads(compact_text)
    assert "m" in compact_payload
    assert "f" in compact_payload
    fact = compact_payload["f"][0]
    assert "b" not in fact
    assert fact["v"] == "9876"
    assert fact["d"] == "30.09.2021"
    assert fact["cur"] == "ILS"
    assert fact["sc"] == 1000
    assert fact["vt"] == "amount"
    assert "ref" in fact
    assert fact["ref"] is None
    assert "note" in fact
    assert fact["note"] is None


def test_export_for_hf_no_bbox_can_resize_images(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_vl_utils(monkeypatch, scale=0.5)

    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Prompt"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": '{"meta":{"type":"other"},"facts":[]}'}]},
        ],
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export_no_bbox"
    train_rows, _ = no_bbox_mod.export_for_hf_no_bbox(
        root,
        export_dir,
        instruction_mode="source",
        max_pixels=100_000,
    )
    assert train_rows == 1

    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    with PILImage.open(export_dir / out["image"]) as img:
        assert img.size == (100, 50)


def test_export_for_hf_no_bbox_excludes_doc_ids(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_vl_utils(monkeypatch, scale=0.5)

    root = tmp_path
    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)

    rows = []
    for doc_id in ("pdf_3", "pdf_4"):
        img_dir = root / "data" / "pdf_images" / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)
        image_path = img_dir / "page_0001.png"
        PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "Prompt"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": '{"meta":{"type":"other"},"facts":[]}'}]},
                ]
            }
        )

    (finetune_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export_no_bbox"
    train_rows, _ = no_bbox_mod.export_for_hf_no_bbox(
        root,
        export_dir,
        instruction_mode="source",
        max_pixels=100_000,
        exclude_doc_ids={"pdf_4"},
    )
    assert train_rows == 1
    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert out["image"].startswith("images/pdf_3/")


def test_build_hf_dataset_no_bbox_from_export_has_instruction_column(tmp_path: Path) -> None:
    export_dir = tmp_path / "artifacts" / "hf_dataset_export_no_bbox"
    img_dir = export_dir / "images" / "doc1"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (20, 20), color=(255, 255, 255)).save(img_path)

    row = {
        "image": "images/doc1/page_0001.png",
        "instruction": no_bbox_mod._MINIMAL_INSTRUCTION,
        "text": '{"meta":{"type":"other"},"facts":[]}',
    }
    (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (export_dir / "val.jsonl").write_text("", encoding="utf-8")

    dataset, train_rows, val_rows = no_bbox_mod.build_hf_dataset_no_bbox_from_export(export_dir, instruction_mode="minimal")
    assert train_rows == 1
    assert val_rows == 0
    assert dataset["train"].column_names == ["image", "instruction", "text"]


def test_main_omit_instruction_alias_maps_to_minimal(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(no_bbox_mod, "build_dataset", lambda *_args, **_kwargs: None)

    def _fake_export(
        root: Path,
        export_dir: Path,
        *,
        instruction_mode: str,
        min_pixels=None,
        max_pixels=None,
        exclude_doc_ids=None,
        compact_tokens: bool = False,
    ):
        _ = root, export_dir, min_pixels, max_pixels, exclude_doc_ids, compact_tokens
        captured["mode"] = instruction_mode
        return 1, 0

    monkeypatch.setattr(no_bbox_mod, "export_for_hf_no_bbox", _fake_export)
    monkeypatch.setattr(no_bbox_mod, "build_hf_dataset_no_bbox_from_export", lambda *_args, **_kwargs: (_empty_dataset(), 1, 0))
    monkeypatch.setattr(no_bbox_mod, "push_to_hf_no_bbox", lambda *_args, **_kwargs: "asafd60/FineTree-annotated-pages-no-bbox-minimal-instruction")

    cwd = Path.cwd()
    try:
        os_root = tmp_path
        (os_root / "configs").mkdir(parents=True)
        (os_root / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
        monkeypatch.chdir(os_root)
        rc = no_bbox_mod.main(["--token", "tok", "--omit-instruction"])
    finally:
        monkeypatch.chdir(cwd)

    assert rc == 0
    assert captured["mode"] == "minimal"
