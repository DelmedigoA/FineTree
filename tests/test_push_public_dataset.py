from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
import pytest

from finetree_annotator.finetune import push_public_dataset as public_mod
from finetree_annotator.schema_contract import PROMPT_PAGE_META_KEYS


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


def test_main_uses_public_trimmed_defaults(monkeypatch, tmp_path: Path) -> None:
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
        _ = (
            allow_format_issues,
            include_doc_ids,
            validation_doc_ids,
            drop_date,
        )
        captured["config_path"] = str(config_path)
        captured["approved_pages_only"] = approved_pages_only
        captured["validation_doc_ids"] = validation_doc_ids
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
        max_pixels: int | None = None,
    ):
        captured["export_root"] = str(root)
        captured["export_dir"] = str(export_dir)
        captured["instruction_mode"] = instruction_mode
        captured["max_pixels"] = max_pixels
        return 11, 2

    def _fake_build_from_export(export_dir: Path, *, instruction_mode: str = "source"):
        captured["build_from_export_dir"] = str(export_dir)
        captured["build_from_export_instruction_mode"] = instruction_mode
        return _empty_dataset(), 11, 2

    def _fake_push(dataset: DatasetDict, token: str, *, base_repo_id: str, private: bool = True):
        _ = dataset
        captured["push_token"] = token
        captured["base_repo_id"] = base_repo_id
        captured["private"] = private
        return {
            "train": f"{base_repo_id}-train",
            "validation": f"{base_repo_id}-validation",
        }

    monkeypatch.setattr(public_mod, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(public_mod, "export_for_hf_no_bbox", _fake_export)
    monkeypatch.setattr(public_mod, "build_hf_dataset_no_bbox_from_export", _fake_build_from_export)
    monkeypatch.setattr(public_mod, "push_train_validation_separately_no_bbox", _fake_push)
    monkeypatch.setattr(public_mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = public_mod.main()

    assert rc == 0
    assert captured["config_path"] == str((tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").resolve())
    assert captured["approved_pages_only"] is False
    assert captured["validation_doc_ids"] == {"test"}
    assert captured["selected_page_meta_keys"] == PROMPT_PAGE_META_KEYS
    assert captured["selected_fact_keys"] == public_mod.PUBLIC_DATASET_FACT_KEYS
    assert captured["page_only_wrapper"] is True
    assert captured["excluded_value_contexts"] == public_mod.PUBLIC_DATASET_EXCLUDED_VALUE_CONTEXTS
    assert captured["include_empty_pages_override"] is True
    assert captured["dedupe_exact_facts"] is True
    assert isinstance(captured["prompt_template_override"], str)
    assert '"bbox"' not in str(captured["prompt_template_override"])
    assert captured["instruction_mode"] == "source"
    assert captured["max_pixels"] == public_mod.DEFAULT_MAX_PIXELS
    assert captured["build_from_export_dir"] == captured["export_dir"]
    assert captured["push_token"] == "hf_test_token"
    assert captured["private"] is False
    assert captured["base_repo_id"] == "user/finetree-v2.5-not-approved"


def test_main_rejects_flags() -> None:
    with pytest.raises(RuntimeError, match="takes no flags"):
        public_mod.main(["--anything"])


def test_main_allows_max_pixels_override_via_env(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv(public_mod.MAX_PIXELS_ENV_VAR, "1200000")

    monkeypatch.setattr(public_mod, "build_dataset", lambda *args, **kwargs: None)

    def _fake_export(
        root: Path,
        export_dir: Path,
        *,
        instruction_mode: str = "source",
        max_pixels: int | None = None,
    ):
        _ = root, export_dir, instruction_mode
        captured["max_pixels"] = max_pixels
        return 1, 1

    monkeypatch.setattr(public_mod, "export_for_hf_no_bbox", _fake_export)
    monkeypatch.setattr(public_mod, "build_hf_dataset_no_bbox_from_export", lambda *args, **kwargs: (_empty_dataset(), 1, 1))
    monkeypatch.setattr(public_mod, "push_train_validation_separately_no_bbox", lambda *args, **kwargs: {"train": "t", "validation": "v"})
    monkeypatch.setattr(public_mod, "HfApi", lambda token: type("FakeApi", (), {"whoami": lambda self: {"name": "user"}})())

    rc = public_mod.main()

    assert rc == 0
    assert captured["max_pixels"] == 1_200_000
