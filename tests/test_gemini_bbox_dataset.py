from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from finetree_annotator.finetune import gemini_bbox_dataset as mod


def _write_config(config_path: Path, annotations_glob: str) -> None:
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  annotations_glob: {annotations_glob}",
                "  images_root: .",
                "  val_ratio: 0.0",
                "  include_empty_pages: true",
                "  bbox_policy: include_if_present",
                "  bbox_space: pixel",
                "  target_schema: finetree_exact_json",
                "  sample_granularity: page",
                "  fact_order_enforce: true",
                "  fact_format_enforce: true",
                "prompt:",
                "  use_custom_prompt: true",
                "  prompt_path: prompts/extraction_prompt.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_annotation(annotation_path: Path, image_path: Path, *, facts: list[dict[str, object]] | None = None) -> None:
    payload = {
        "schema_version": 3,
        "images_dir": str(image_path.parent),
        "metadata": {"language": "en", "reading_direction": "ltr"},
        "pages": [
            {
                "image": image_path.name,
                "meta": {
                    "entity_name": "Demo Co",
                    "page_num": "1",
                    "page_type": "statements",
                    "statement_type": "income_statement",
                    "title": "Income Statement",
                },
                "facts": facts
                if facts is not None
                else [
                    {
                        "bbox": [100, 50, 20, 10],
                        "value": "10",
                        "fact_num": 1,
                        "path": ["Revenue"],
                        "note_flag": False,
                        "value_context": "tabular",
                    }
                ],
            }
        ],
    }
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_convert_bbox_to_gemini_1000_from_pixel_xywh() -> None:
    bbox, clamped, source_format = mod.convert_bbox_to_gemini_1000(
        [100, 50, 20, 10],
        image_width=200,
        image_height=100,
        source_format="pixel_xywh",
    )

    assert bbox == [500, 500, 600, 600]
    assert clamped is False
    assert source_format == "pixel_xywh"


def test_validate_gemini_bbox_1000_rejects_non_integer_box() -> None:
    with pytest.raises(ValueError, match="integers"):
        mod.validate_gemini_bbox_1000([100, 200, 300.5, 400])


def test_validate_gemini_bbox_1000_rejects_odd_box() -> None:
    with pytest.raises(ValueError, match="even integers"):
        mod.validate_gemini_bbox_1000([100, 201, 300, 400])


def test_convert_bbox_to_gemini_1000_snaps_to_even_outward_edges() -> None:
    bbox, clamped, source_format = mod.convert_bbox_to_gemini_1000(
        [1, 1, 2, 2],
        image_width=7,
        image_height=7,
        source_format="pixel_xywh",
    )

    assert bbox == [142, 142, 430, 430]
    assert clamped is False
    assert source_format == "pixel_xywh"


def test_build_gemini_bbox_prompt_template_is_schema_first() -> None:
    prompt = mod.build_gemini_bbox_prompt_template()

    assert "Current image size: {{IMAGE_DIMENSIONS}}." in prompt
    assert "Current page image:" not in prompt
    assert "Exact schema:" in prompt
    assert '"meta": {' in prompt
    assert '"facts": [' in prompt
    assert '"bbox": [<ymin>, <xmin>, <ymax>, <xmax>]' in prompt


def test_prepare_gemini_bbox_dataset_writes_normalized_boxes(tmp_path: Path) -> None:
    images_dir = tmp_path / "images_src"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "page_0001.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True)
    annotation_path = annotations_dir / "doc_a.json"
    _write_annotation(annotation_path, image_path)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, str((annotations_dir / "*.json").resolve()))

    export_dir = tmp_path / "export"
    stats = mod.prepare_gemini_bbox_dataset(
        root=tmp_path,
        config_path=config_path,
        export_dir=export_dir,
        allow_format_issues=True,
        preview_count=1,
    )

    assert stats.train_rows == 1
    assert stats.preview_images_created == 1
    train_lines = [line for line in (export_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_lines) == 1
    row = json.loads(train_lines[0])
    assert row["image"] == "images/doc_a/page_0001.png"
    assert "[ymin, xmin, ymax, xmax]" in row["instruction"]
    assert "Current image size: 200 x 100 pixels." in row["instruction"]
    assert "Current page image:" not in row["instruction"]
    assert '"bbox": [<ymin>, <xmin>, <ymax>, <xmax>]' in row["instruction"]
    assert '"meta": {' in row["instruction"]
    assert '"facts": [' in row["instruction"]

    payload = json.loads(row["text"])
    assert payload["facts"][0]["bbox"] == [500, 500, 600, 600]
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bbox_space"] == "normalized_1000_integer"
    assert manifest["bbox_quantization"] == "even_outward_safe"
    preview_summary = json.loads((export_dir / "bbox_preview_samples" / "summary.json").read_text(encoding="utf-8"))
    assert preview_summary["created"] == 1
    preview_rel = preview_summary["samples"][0]["preview_image"]
    assert (export_dir / preview_rel).is_file()


def test_build_hf_dataset_from_gemini_export_reads_export_dir(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    (export_dir / "images" / "doc_a").mkdir(parents=True)
    image_path = export_dir / "images" / "doc_a" / "page.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(image_path)
    row = {
        "image": "images/doc_a/page.png",
        "system": "sys",
        "instruction": mod.MINIMAL_GEMINI_BBOX_INSTRUCTION,
        "text": json.dumps(
            {
                "meta": {
                    "entity_name": None,
                    "page_num": "1",
                    "page_type": "statements",
                    "statement_type": "income_statement",
                    "title": None,
                },
                "facts": [
                    {
                        "bbox": [100, 200, 300, 400],
                        "value": "10",
                        "fact_num": 1,
                        "path": ["Revenue"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
    }
    (export_dir / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (export_dir / "val.jsonl").write_text("", encoding="utf-8")

    dataset, train_rows, val_rows = mod.build_hf_dataset_from_gemini_export(export_dir)

    assert train_rows == 1
    assert val_rows == 0
    assert len(dataset["train"]) == 1


def test_prepare_gemini_bbox_dataset_removes_duplicates_when_requested(tmp_path: Path) -> None:
    images_dir = tmp_path / "images_src"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "page_0001.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True)
    annotation_path = annotations_dir / "doc_dup.json"
    duplicate_fact = {
        "bbox": [100, 50, 20, 10],
        "value": "10",
        "fact_num": 1,
        "path": ["Revenue"],
        "note_flag": False,
        "value_context": "tabular",
    }
    _write_annotation(annotation_path, image_path, facts=[duplicate_fact, dict(duplicate_fact)])

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, str((annotations_dir / "*.json").resolve()))

    export_dir = tmp_path / "export"
    stats = mod.prepare_gemini_bbox_dataset(
        root=tmp_path,
        config_path=config_path,
        export_dir=export_dir,
        allow_format_issues=True,
        remove_duplicates=True,
        preview_count=1,
    )

    row = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    payload = json.loads(row["text"])
    assert stats.facts_deduped == 1
    assert len(payload["facts"]) == 1
    assert payload["facts"][0]["bbox"] == [500, 500, 600, 600]


def test_prepare_gemini_bbox_dataset_filters_textual_facts_but_keeps_empty_page(tmp_path: Path) -> None:
    images_dir = tmp_path / "images_src"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "page_0001.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True)
    annotation_path = annotations_dir / "doc_textual.json"
    _write_annotation(
        annotation_path,
        image_path,
        facts=[
            {
                "bbox": [100, 50, 20, 10],
                "value": "10",
                "fact_num": 1,
                "path": ["Narrative"],
                "note_flag": False,
                "value_context": "textual",
            }
        ],
    )

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, str((annotations_dir / "*.json").resolve()))

    export_dir = tmp_path / "export"
    stats = mod.prepare_gemini_bbox_dataset(
        root=tmp_path,
        config_path=config_path,
        export_dir=export_dir,
        allow_format_issues=True,
        preview_count=1,
    )

    assert stats.train_rows == 1
    assert stats.preview_images_created == 0
    row = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    payload = json.loads(row["text"])
    assert payload["facts"] == []
    preview_summary = json.loads((export_dir / "bbox_preview_samples" / "summary.json").read_text(encoding="utf-8"))
    assert preview_summary["created"] == 0


def test_main_push_prepares_then_pushes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    called: dict[str, object] = {}

    def _fake_prepare(**kwargs):
        called["prepare"] = kwargs
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "train.jsonl").write_text("", encoding="utf-8")
        (export_dir / "val.jsonl").write_text("", encoding="utf-8")
        return mod.GeminiBBoxExportStats()

    def _fake_build(path: Path):
        called["build"] = path
        return {"train": ["train_row"], "validation": ["val_row"]}, 1, 1

    def _fake_push(dataset, token: str, *, repo_id: str, private: bool):
        called["push"] = {
            "dataset": dataset,
            "token": token,
            "repo_id": repo_id,
            "private": private,
        }
        return "user/gemini-bbox"

    monkeypatch.setattr(mod, "prepare_gemini_bbox_dataset", _fake_prepare)
    monkeypatch.setattr(mod, "build_hf_dataset_from_gemini_export", _fake_build)
    monkeypatch.setattr(mod, "push_gemini_bbox_dataset_to_hf", _fake_push)
    monkeypatch.setattr(mod, "resolve_hf_token", lambda _token: "hf_test")

    exit_code = mod.main_push(
        [
            "--repo-id",
            "user/gemini-bbox",
            "--export-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    assert "prepare" in called
    assert called["build"] == export_dir.resolve()
    assert called["push"]["repo_id"] == "user/gemini-bbox"
    assert called["push"]["private"] is False
    assert set(called["push"]["dataset"].keys()) == {"train", "validation"}


def test_parse_push_args_rejects_split_repo_overrides() -> None:
    with pytest.raises(SystemExit):
        mod.parse_push_args(
            [
                "--repo-id",
                "user/gemini-bbox",
                "--repo-id-train",
                "user/gemini-bbox-train",
            ]
        )
