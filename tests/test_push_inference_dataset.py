from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from finetree_annotator.finetune import push_inference_dataset as inference_mod


def _empty_dataset() -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "system": Value("string"),
            "instruction": Value("string"),
            "doc_id": Value("string"),
            "page_image": Value("string"),
        }
    )
    empty = {"image": [], "system": [], "instruction": [], "doc_id": [], "page_image": []}
    return DatasetDict({"train": Dataset.from_dict(empty, features=features)})


def test_inference_dataset_name_uses_stable_four_char_short_id() -> None:
    name = inference_mod.inference_dataset_name("pdf_english", 1400000)
    assert name.startswith("pdf_")
    assert name.endswith("_for_inference_1400000")
    short_id = name[len("pdf_") : len("pdf_") + 4]
    assert len(short_id) == 4
    assert short_id == inference_mod.inference_dataset_short_id("pdf_english")


def test_export_pdf_for_inference_writes_rows_and_resized_images(tmp_path: Path) -> None:
    root = tmp_path
    images_dir = root / "data" / "pdf_images" / "doc1"
    images_dir.mkdir(parents=True)
    PILImage.new("RGB", (400, 200), color=(255, 255, 255)).save(images_dir / "page_0001.png")
    PILImage.new("RGB", (400, 200), color=(240, 240, 240)).save(images_dir / "page_0002.png")
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "system_prompt.txt").write_text("System prompt from file", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_inference_export" / "doc1"
    row_count = inference_mod.export_pdf_for_inference(
        root,
        doc_id="doc1",
        images_dir=images_dir,
        export_dir=export_dir,
        max_pixels=10_000,
        page_meta_keys=(),
        fact_keys=("value", "path"),
    )

    assert row_count == 2
    rows = [
        line
        for line in (export_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    first = json.loads(rows[0])
    assert first["system"] == "System prompt from file"
    assert first["doc_id"] == "doc1"
    assert first["page_image"] == "page_0001.png"
    assert "Selected page meta keys:" in first["instruction"]
    assert "- (none)" in first["instruction"]
    assert "Selected fact keys:" in first["instruction"]
    assert "- value, path" in first["instruction"]
    assert '"bbox"' not in first["instruction"]
    assert "single compact line" in first["instruction"]

    with PILImage.open(export_dir / first["image"]) as exported:
        assert exported.width * exported.height <= 10_000

    dataset, dataset_rows = inference_mod.build_hf_inference_dataset_from_export(export_dir)
    assert dataset_rows == 2
    assert list(dataset.keys()) == ["train"]
    assert len(dataset["train"]) == 2


def test_main_uses_requested_name_and_selected_schema(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    images_dir = tmp_path / "data" / "pdf_images" / "doc_alpha"
    images_dir.mkdir(parents=True)
    PILImage.new("RGB", (32, 32), color=(255, 255, 255)).save(images_dir / "page_0001.png")

    def _fake_export(
        root: Path,
        *,
        doc_id: str,
        images_dir: Path,
        export_dir: Path,
        max_pixels: int,
        page_meta_keys,
        fact_keys,
    ) -> int:
        captured["root"] = str(root)
        captured["doc_id"] = doc_id
        captured["images_dir"] = str(images_dir)
        captured["export_dir"] = str(export_dir)
        captured["max_pixels"] = max_pixels
        captured["page_meta_keys"] = page_meta_keys
        captured["fact_keys"] = fact_keys
        return 7

    def _fake_build(export_dir: Path):
        captured["build_export_dir"] = str(export_dir)
        return _empty_dataset(), 7

    def _fake_push(dataset: DatasetDict, *, token: str, doc_id: str, max_pixels: int, repo_id=None, private: bool = False) -> str:
        _ = dataset, repo_id
        captured["push_token"] = token
        captured["push_doc_id"] = doc_id
        captured["push_max_pixels"] = max_pixels
        captured["private"] = private
        return f"user/{inference_mod.inference_dataset_name(doc_id, max_pixels)}"

    monkeypatch.setattr(inference_mod, "export_pdf_for_inference", _fake_export)
    monkeypatch.setattr(inference_mod, "build_hf_inference_dataset_from_export", _fake_build)
    monkeypatch.setattr(inference_mod, "push_inference_dataset", _fake_push)

    rc = inference_mod.main(
        [
            "--doc-id",
            "doc_alpha",
            "--images-dir",
            str(images_dir),
            "--max-pixels",
            "1200000",
            "--page-meta-keys",
            "page_num,title",
            "--fact-keys",
            "value,path,note_ref",
        ]
    )

    assert rc == 0
    assert captured["doc_id"] == "doc_alpha"
    assert captured["max_pixels"] == 1_200_000
    assert captured["page_meta_keys"] == ("page_num", "title")
    assert captured["fact_keys"] == ("value", "path", "note_ref")
    assert captured["build_export_dir"] == captured["export_dir"]
    assert captured["push_token"] == "hf_test_token"
    assert captured["push_doc_id"] == "doc_alpha"
    assert captured["push_max_pixels"] == 1_200_000
    assert captured["private"] is False
    expected_name = inference_mod.inference_dataset_name("doc_alpha", 1_200_000)
    assert str(captured["export_dir"]).endswith(f"/artifacts/hf_inference_export/{expected_name}")


def test_build_inference_instruction_defaults_to_requested_no_bbox_schema() -> None:
    instruction = inference_mod.build_inference_instruction()

    assert "entity_name, page_num, page_type, statement_type, title" in instruction
    assert (
        "value, fact_num, comment_ref, note_flag, note_name, note_num, note_ref, "
        "period_type, period_start, period_end, path, path_source, currency, scale, value_type, value_context"
    ) in instruction
    assert '"bbox"' not in instruction
    assert "single compact line" in instruction
    assert "Do not pretty-print" in instruction
