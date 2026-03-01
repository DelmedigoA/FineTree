from __future__ import annotations

from finetree_annotator.inference.model_family import canonical_model_id, is_qwen35_a3_model


def test_canonical_model_id_normalizes_separators_and_case() -> None:
    assert canonical_model_id(" Unsloth/Qwen3.5-35B-A3B ") == "unslothqwen3535ba3b"
    assert canonical_model_id("Qwen/QWEN3_5_A3B") == "qwenqwen35a3b"


def test_is_qwen35_a3_model_accepts_multiple_repo_forms() -> None:
    assert is_qwen35_a3_model("unsloth/Qwen3.5-35B-A3B") is True
    assert is_qwen35_a3_model("org/qwen3_5-30b-a3b-instruct") is True
    assert is_qwen35_a3_model("Qwen3.5 A3 custom") is True


def test_is_qwen35_a3_model_rejects_other_families() -> None:
    assert is_qwen35_a3_model("Qwen/Qwen2.5-7B-Instruct") is False
    assert is_qwen35_a3_model("meta-llama/Llama-3.1-8B") is False
    assert is_qwen35_a3_model(None) is False
