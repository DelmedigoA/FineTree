from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "finetree-v3-factnum-doctr.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("finetree_v3_factnum_doctr", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_xywh_pixel_to_polygon_abs_uses_pixel_xywh() -> None:
    mod = _load_script_module()

    polygon = mod.xywh_pixel_to_polygon_abs([10, 20, 30, 40], width=200, height=100)

    assert polygon == [[10, 20], [40, 20], [40, 60], [10, 60]]


def test_build_doctr_split_writes_labels_and_skips_empty(tmp_path: Path) -> None:
    mod = _load_script_module()
    image = Image.new("RGB", (100, 80), color="white")
    rows = [
        {
            "image": image,
            "text": json.dumps(
                {
                    "facts": [
                        {"bbox": [10, 10, 30, 20], "value": "1", "fact_num": 1},
                        {"bbox": [10, 10, 30, 20], "value": "1", "fact_num": 1},
                    ]
                }
            ),
        },
        {
            "image": image,
            "text": json.dumps({"facts": []}),
        },
    ]

    stats = mod.build_doctr_split(rows, tmp_path / "train")

    assert stats == {
        "processed_rows": 2,
        "kept_images": 1,
        "skipped_empty": 1,
        "polygons": 1,
    }
    labels = json.loads((tmp_path / "train" / "labels.json").read_text(encoding="utf-8"))
    assert sorted(labels.keys()) == ["00000.png"]
    assert labels["00000.png"]["img_dimensions"] == [80, 100]
    assert labels["00000.png"]["polygons"] == [[[10, 10], [40, 10], [40, 30], [10, 30]]]
    assert (tmp_path / "train" / "images" / "00000.png").is_file()


def test_main_dry_run_exports_train_and_val(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_script_module()

    monkeypatch.chdir(tmp_path)
    image = Image.new("RGB", (64, 32), color="white")

    def _fake_load_dataset(repo_id: str, *, split: str):
        assert repo_id == "asafd60/fintetree-v3-factnum"
        if split == "train":
            return [
                {"image": image, "text": json.dumps({"facts": [{"bbox": [1, 2, 3, 4]}]})},
                {"image": image, "text": json.dumps({"facts": []})},
            ]
        if split == "validation":
            return [{"image": image, "text": json.dumps({"facts": [{"bbox": [2, 3, 4, 5]}]})}]
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(mod, "load_dataset", _fake_load_dataset)

    rc = mod.main(["--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "SOURCE_DATASET: asafd60/fintetree-v3-factnum" in out
    assert "TRAIN_ROWS: 2" in out
    assert "TRAIN_IMAGES: 1" in out
    assert "TRAIN_SKIPPED_EMPTY: 1" in out
    assert "VAL_ROWS: 1" in out
    assert "VAL_IMAGES: 1" in out
    assert "REPO_ID: asafd60/fintetree-v3-factnum-doctr" in out
    assert "PUBLIC: True" in out
    assert "DRY_RUN: true" in out
    assert (tmp_path / "artifacts" / "fintetree_v3_factnum_doctr" / "train" / "labels.json").is_file()
    assert (tmp_path / "artifacts" / "fintetree_v3_factnum_doctr" / "val" / "labels.json").is_file()
