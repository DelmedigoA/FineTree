from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from datasets import Dataset, DatasetDict, Features, Image, Value


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "finetree-v2_8.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("finetree_v2_8", SCRIPT_PATH)
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


def test_normalize_export_value_examples() -> None:
    mod = _load_script_module()

    assert mod._normalize_export_value("1000") == "1,000"
    assert mod._normalize_export_value("-1000") == "-1,000"
    assert mod._normalize_export_value("- 1000") == "-1,000"
    assert mod._normalize_export_value("1234.56") == "1,234.56"
    assert mod._normalize_export_value("(1234)") == "(1,234)"

    assert mod._normalize_export_value("-") == "-"
    assert mod._normalize_export_value("7-15") == "7-15"
    assert mod._normalize_export_value("12.5%") == "12.5%"
    assert mod._normalize_export_value("*62,565") == "*62,565"


def test_build_prompt_template_requires_compact_single_line_json() -> None:
    mod = _load_script_module()

    prompt = mod._build_prompt_template()

    assert '"bbox"' not in prompt
    assert "single compact line" in prompt
    assert "Do not pretty-print" in prompt
    assert "Do not add any prefix, suffix, explanation, or surrounding text." in prompt


def test_main_dry_run_builds_exact_schema_and_normalizes_export(monkeypatch, tmp_path: Path, capsys) -> None:
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
        captured["excluded_value_contexts"] = excluded_value_contexts
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
    ):
        _ = root, min_pixels, exclude_doc_ids, compact_tokens, aggressive_compact_tokens
        captured["export_dir"] = str(export_dir)
        captured["instruction_mode"] = instruction_mode
        captured["max_pixels"] = max_pixels
        export_dir.mkdir(parents=True, exist_ok=True)
        train_payload = {
            "meta": {"entity_name": None, "page_num": None, "page_type": "other", "statement_type": None, "title": None},
            "facts": [
                {"value": "1000", "path": []},
                {"value": "7-15", "path": []},
            ],
        }
        val_payload = {
            "meta": {"entity_name": None, "page_num": None, "page_type": "other", "statement_type": None, "title": None},
            "facts": [
                {"value": "-1000", "path": []},
                {"value": "*62,565", "path": []},
            ],
        }
        train_row = {"image": "images/doc1/page.png", "system": "s", "instruction": "i", "text": json.dumps(train_payload)}
        val_row = {"image": "images/doc2/page.png", "system": "s", "instruction": "i", "text": json.dumps(val_payload)}
        (export_dir / "train.jsonl").write_text(json.dumps(train_row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text(json.dumps(val_row) + "\n", encoding="utf-8")
        return 2, 2

    def _fake_build_from_export(export_dir: Path, *, instruction_mode: str = "source"):
        captured["build_from_export_dir"] = str(export_dir)
        captured["build_from_export_instruction_mode"] = instruction_mode
        train_line = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
        val_line = json.loads((export_dir / "val.jsonl").read_text(encoding="utf-8").strip())
        captured["train_values"] = [fact["value"] for fact in json.loads(train_line["text"])["facts"]]
        captured["val_values"] = [fact["value"] for fact in json.loads(val_line["text"])["facts"]]
        return _empty_dataset(), 2, 2

    def _fake_push(*args, **kwargs):
        raise AssertionError("push should not be called during --dry-run")

    monkeypatch.setattr(mod, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(mod, "export_for_hf_no_bbox", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_no_bbox_from_export", _fake_build_from_export)
    monkeypatch.setattr(mod, "push_train_validation_separately_no_bbox", _fake_push)
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = mod.main(["--dry-run"])

    assert rc == 0
    assert captured["config_path"] == str((tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").resolve())
    assert captured["validation_doc_ids"] == {"pdf_4"}
    assert captured["approved_pages_only"] is True
    assert captured["selected_page_meta_keys"] == mod.DEFAULT_PAGE_META_KEYS
    assert captured["selected_fact_keys"] == mod.DEFAULT_FACT_KEYS
    assert captured["page_only_wrapper"] is True
    assert captured["excluded_value_contexts"] is None
    assert captured["include_empty_pages_override"] is True
    assert captured["dedupe_exact_facts"] is True
    assert '"bbox"' not in str(captured["prompt_template_override"])
    assert captured["instruction_mode"] == "source"
    assert captured["max_pixels"] == mod.DEFAULT_MAX_PIXELS
    assert captured["train_values"] == ["1,000", "7-15"]
    assert captured["val_values"] == ["-1,000", "*62,565"]

    out = capsys.readouterr().out
    assert "DRY_RUN: true" in out
    assert "REPO_ID_TRAIN: user/FineTree-2.8-train" in out
    assert "REPO_ID_VALIDATION: user/FineTree-2.8-validation" in out
    assert "EXCLUDED_VALUE_CONTEXTS: []" in out
    assert "NORMALIZED_VALUES: 2" in out
    assert "UNCHANGED_VALUES: 2" in out


def test_main_pushes_public_split_repos(monkeypatch, tmp_path: Path) -> None:
    mod = _load_script_module()
    captured: dict[str, object] = {}

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
            "text": json.dumps({"meta": {}, "facts": [{"value": "1000", "path": []}]}),
        }
        (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text("", encoding="utf-8")
        return 1, 0

    def _fake_push(dataset, token: str, *, base_repo_id: str, private: bool = False, repo_id_train: str | None = None, repo_id_validation: str | None = None):
        _ = dataset, base_repo_id
        captured["token"] = token
        captured["private"] = private
        captured["repo_id_train"] = repo_id_train
        captured["repo_id_validation"] = repo_id_validation
        return {"train": repo_id_train, "validation": repo_id_validation}

    monkeypatch.setattr(mod, "export_for_hf_no_bbox", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_no_bbox_from_export", lambda *args, **kwargs: (_empty_dataset(), 1, 0))
    monkeypatch.setattr(mod, "push_train_validation_separately_no_bbox", _fake_push)
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = mod.main([])

    assert rc == 0
    assert captured["token"] == "hf_test_token"
    assert captured["private"] is False
    assert captured["repo_id_train"] == "user/FineTree-2.8-train"
    assert captured["repo_id_validation"] == "user/FineTree-2.8-validation"


def test_main_include_tabular_mixed_only_excludes_textual(monkeypatch, tmp_path: Path) -> None:
    mod = _load_script_module()
    captured: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    def _fake_build_dataset(config_path: Path, **kwargs) -> None:
        _ = config_path
        captured["excluded_value_contexts"] = kwargs.get("excluded_value_contexts")

    monkeypatch.setattr(mod, "build_dataset", _fake_build_dataset)

    def _fake_export(root: Path, export_dir: Path, **kwargs):
        _ = root, kwargs
        export_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "images/doc1/page.png",
            "system": "s",
            "instruction": "i",
            "text": json.dumps({"meta": {}, "facts": []}),
        }
        (export_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (export_dir / "val.jsonl").write_text("", encoding="utf-8")
        return 0, 0

    monkeypatch.setattr(mod, "export_for_hf_no_bbox", _fake_export)
    monkeypatch.setattr(mod, "build_hf_dataset_no_bbox_from_export", lambda *args, **kwargs: (_empty_dataset(), 0, 0))
    monkeypatch.setattr(mod, "push_train_validation_separately_no_bbox", lambda *args, **kwargs: {"train": "t", "validation": "v"})
    monkeypatch.setattr(mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = mod.main(["--dry-run", "--include_tabular_mixed_only"])

    assert rc == 0
    assert captured["excluded_value_contexts"] == ("textual",)
