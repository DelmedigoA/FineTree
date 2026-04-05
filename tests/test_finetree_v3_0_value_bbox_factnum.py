from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from datasets import Dataset, DatasetDict, Features, Image, Value


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "finetree-v3_0-value-bbox-factnum.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("finetree_v3_0_value_bbox_factnum", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _empty_dataset() -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "system": Value("string"),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    empty = {"image": [], "system": [], "instruction": [], "text": []}
    return DatasetDict(
        {
            "train": Dataset.from_dict(empty, features=features),
            "validation": Dataset.from_dict(empty, features=features),
        }
    )


def _dataset_with_counts(train_count: int, validation_count: int) -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "system": Value("string"),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )

    def _rows(prefix: str, count: int) -> Dataset:
        rows = [
            {
                "image": f"/tmp/{prefix}_{idx}.png",
                "system": "s",
                "instruction": "i",
                "text": json.dumps({"meta": {}, "facts": [{"bbox": [1, 2, 3, 4], "value": str(idx), "fact_num": idx}]}),
            }
            for idx in range(count)
        ]
        if rows:
            return Dataset.from_list(rows, features=features)
        empty = {"image": [], "system": [], "instruction": [], "text": []}
        return Dataset.from_dict(empty, features=features)

    return DatasetDict(
        {
            "train": _rows("train", train_count),
            "validation": _rows("validation", validation_count),
        }
    )


def test_build_prompt_template_keeps_bbox_and_compact_output() -> None:
    mod = _load_script_module()

    prompt = mod._build_prompt_template()

    assert '"bbox"' in prompt
    assert "single compact line" in prompt
    assert "Do not pretty-print" in prompt


def test_main_dry_run_builds_bbox_dataset_with_selected_fact_keys(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_script_module()
    captured: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    def _fake_build_dataset(
        config_path: Path,
        *,
        allow_format_issues: bool = False,
        include_doc_ids=None,
        validation_doc_ids=None,
        approved_pages_only: bool = False,
        drop_date: bool = False,
        prompt_template_override: str | None = None,
        selected_page_meta_keys=None,
        selected_fact_keys=None,
        page_only_wrapper: bool = False,
        excluded_value_contexts=None,
        include_empty_pages_override: bool | None = None,
        dedupe_exact_facts: bool = False,
    ) -> None:
        _ = allow_format_issues, include_doc_ids, drop_date, excluded_value_contexts
        captured["config_path"] = str(config_path)
        captured["validation_doc_ids"] = validation_doc_ids
        captured["approved_pages_only"] = approved_pages_only
        captured["prompt_template_override"] = prompt_template_override
        captured["selected_page_meta_keys"] = selected_page_meta_keys
        captured["selected_fact_keys"] = selected_fact_keys
        captured["page_only_wrapper"] = page_only_wrapper
        captured["include_empty_pages_override"] = include_empty_pages_override
        captured["dedupe_exact_facts"] = dedupe_exact_facts

    def _fake_export(
        root: Path,
        export_dir: Path,
        *,
        instruction_mode: str = "source",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        exclude_doc_ids=None,
        compact_tokens: bool = False,
        aggressive_compact_tokens: bool = False,
        drop_date: bool = False,
    ):
        _ = root, min_pixels, exclude_doc_ids, compact_tokens, aggressive_compact_tokens, drop_date
        captured["export_dir"] = str(export_dir)
        captured["instruction_mode"] = instruction_mode
        captured["max_pixels"] = max_pixels
        export_dir.mkdir(parents=True, exist_ok=True)
        train_payload = {
            "meta": {"entity_name": None, "page_num": None, "page_type": "other", "statement_type": None, "title": None},
            "facts": [
                {"bbox": [1, 2, 3, 4], "value": "1000", "fact_num": 1},
                {"bbox": [5, 6, 7, 8], "value": "7-15", "fact_num": 2},
            ],
        }
        val_payload = {
            "meta": {"entity_name": None, "page_num": None, "page_type": "other", "statement_type": None, "title": None},
            "facts": [
                {"bbox": [1, 2, 3, 4], "value": "-1000", "fact_num": 1},
                {"bbox": [5, 6, 7, 8], "value": "*62,565", "fact_num": 2},
            ],
        }
        train_row = {"image": "images/doc1/page.png", "system": "s", "instruction": "i", "text": json.dumps(train_payload)}
        val_row = {"image": "images/doc2/page.png", "system": "s", "instruction": "i", "text": json.dumps(val_payload)}
        (export_dir / "train.jsonl").write_text(json.dumps(train_row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text(json.dumps(val_row) + "\n", encoding="utf-8")
        return 2, 2

    def _fake_build_from_export(export_dir: Path):
        captured["build_from_export_dir"] = str(export_dir)
        train_line = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
        val_line = json.loads((export_dir / "val.jsonl").read_text(encoding="utf-8").strip())
        captured["train_values"] = [fact["value"] for fact in json.loads(train_line["text"])["facts"]]
        captured["val_values"] = [fact["value"] for fact in json.loads(val_line["text"])["facts"]]
        captured["train_bbox"] = [fact["bbox"] for fact in json.loads(train_line["text"])["facts"]]
        return _empty_dataset(), 2, 2

    def _fake_push(*args, **kwargs):
        raise AssertionError("push should not be called during --dry-run")

    monkeypatch.setattr(mod, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(mod, "export_for_hf", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_from_export", _fake_build_from_export)
    monkeypatch.setattr(mod, "push_train_validation_separately", _fake_push)
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = mod.main(["--dry-run"])

    assert rc == 0
    assert captured["config_path"] == str((tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").resolve())
    assert captured["validation_doc_ids"] == {"pdf_4"}
    assert captured["approved_pages_only"] is True
    assert captured["selected_page_meta_keys"] == mod.DEFAULT_PAGE_META_KEYS
    assert captured["selected_fact_keys"] == mod.DEFAULT_FACT_KEYS
    assert captured["page_only_wrapper"] is True
    assert captured["include_empty_pages_override"] is True
    assert captured["dedupe_exact_facts"] is True
    assert '"bbox"' in str(captured["prompt_template_override"])
    assert captured["instruction_mode"] == "source"
    assert captured["max_pixels"] == mod.DEFAULT_MAX_PIXELS
    assert captured["train_values"] == ["1,000", "7-15"]
    assert captured["val_values"] == ["-1,000", "*62,565"]
    assert captured["train_bbox"] == [[1, 2, 3, 4], [5, 6, 7, 8]]

    out = capsys.readouterr().out
    assert "DRY_RUN: true" in out
    assert "REPO_ID_TRAIN: user/FineTree-3.0-value-bbox-factnum-train" in out
    assert "REPO_ID_VALIDATION: user/FineTree-3.0-value-bbox-factnum-validation" in out
    assert "INCLUDE_BBOX: true" in out
    assert "NORMALIZED_VALUES: 2" in out
    assert "UNCHANGED_VALUES: 2" in out


def test_main_dry_run_full_resolution_disables_resize(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_script_module()
    captured: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    monkeypatch.setattr(mod, "build_dataset", lambda *args, **kwargs: None)

    def _fake_export(root: Path, export_dir: Path, **kwargs):
        _ = root, export_dir
        captured["max_pixels"] = kwargs.get("max_pixels")
        export_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "images/doc1/page.png",
            "system": "s",
            "instruction": "i",
            "text": json.dumps({"meta": {}, "facts": [{"bbox": [1, 2, 3, 4], "value": "1000", "fact_num": 1}]}),
        }
        (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text("", encoding="utf-8")
        return 1, 0

    monkeypatch.setattr(mod, "export_for_hf", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_from_export", lambda *args, **kwargs: (_empty_dataset(), 1, 0))
    monkeypatch.setattr(mod, "push_train_validation_separately", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = mod.main(["--dry-run", "--full-resolution"])

    assert rc == 0
    assert captured["max_pixels"] is None

    out = capsys.readouterr().out
    assert "MAX_PIXELS: unset" in out
    assert "FULL_RESOLUTION: True" in out


def test_main_dry_run_merges_existing_hf_dataset_by_split(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_script_module()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    monkeypatch.setattr(mod, "build_dataset", lambda *args, **kwargs: None)

    def _fake_export(root: Path, export_dir: Path, **kwargs):
        _ = root, kwargs
        export_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "images/doc1/page.png",
            "system": "s",
            "instruction": "i",
            "text": json.dumps({"meta": {}, "facts": [{"bbox": [1, 2, 3, 4], "value": "1000", "fact_num": 1}]}),
        }
        (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return 1, 1

    monkeypatch.setattr(mod, "export_for_hf", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_from_export", lambda *args, **kwargs: (_dataset_with_counts(2, 1), 2, 1))
    monkeypatch.setattr(mod, "push_train_validation_separately", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    def _fake_load_dataset(repo_id: str, *, split: str):
        assert repo_id == "asafd60/existing-facts"
        if split == "train":
            return _dataset_with_counts(3, 0)["train"]
        if split == "validation":
            return _dataset_with_counts(0, 4)["validation"]
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(mod, "load_dataset", _fake_load_dataset)

    rc = mod.main(["--dry-run", "--merge-hf-dataset", "asafd60/existing-facts"])

    assert rc == 0

    out = capsys.readouterr().out
    assert "MERGE_HF_DATASET: asafd60/existing-facts" in out
    assert "MERGE_TRAIN_SPLIT: train" in out
    assert "MERGE_VALIDATION_SPLIT: validation" in out
    assert "LOCAL_TRAIN_ROWS: 2" in out
    assert "LOCAL_VAL_ROWS: 1" in out
    assert "MERGED_HF_TRAIN_ROWS: 3" in out
    assert "MERGED_HF_VAL_ROWS: 4" in out
    assert "FINAL_PUSH_TRAIN_ROWS: 5" in out
    assert "FINAL_PUSH_VAL_ROWS: 5" in out


def test_main_dry_run_uses_base_repo_id_override(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_script_module()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    monkeypatch.setattr(mod, "build_dataset", lambda *args, **kwargs: None)

    def _fake_export(root: Path, export_dir: Path, **kwargs):
        _ = root, kwargs
        export_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "images/doc1/page.png",
            "system": "s",
            "instruction": "i",
            "text": json.dumps({"meta": {}, "facts": [{"bbox": [1, 2, 3, 4], "value": "1000", "fact_num": 1}]}),
        }
        (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text("", encoding="utf-8")
        return 1, 0

    monkeypatch.setattr(mod, "export_for_hf", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_from_export", lambda *args, **kwargs: (_empty_dataset(), 1, 0))
    monkeypatch.setattr(mod, "push_train_validation_separately", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "ignored-user"}})())

    rc = mod.main(["--dry-run", "--base-repo-id", "asafd60/fintetree-v3-factnum"])

    assert rc == 0

    out = capsys.readouterr().out
    assert "REPO_ID_TRAIN: asafd60/fintetree-v3-factnum-train" in out
    assert "REPO_ID_VALIDATION: asafd60/fintetree-v3-factnum-validation" in out


def test_main_dry_run_reports_single_repo_override(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_script_module()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    monkeypatch.setattr(mod, "build_dataset", lambda *args, **kwargs: None)

    def _fake_export(root: Path, export_dir: Path, **kwargs):
        _ = root, kwargs
        export_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "images/doc1/page.png",
            "system": "s",
            "instruction": "i",
            "text": json.dumps({"meta": {}, "facts": [{"bbox": [1, 2, 3, 4], "value": "1000", "fact_num": 1}]}),
        }
        (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return 1, 1

    monkeypatch.setattr(mod, "export_for_hf", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_from_export", lambda *args, **kwargs: (_dataset_with_counts(1, 1), 1, 1))
    monkeypatch.setattr(mod, "push_train_validation_separately", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "push_to_hf", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "ignored-user"}})())

    rc = mod.main(["--dry-run", "--repo-id", "asafd60/fintetree-v3-factnum"])

    assert rc == 0

    out = capsys.readouterr().out
    assert "REPO_ID: asafd60/fintetree-v3-factnum" in out
    assert "REPO_ID_TRAIN: ignored-user/FineTree-3.0-value-bbox-factnum-train" in out
    assert "REPO_ID_VALIDATION: ignored-user/FineTree-3.0-value-bbox-factnum-validation" in out
