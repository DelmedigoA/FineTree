from __future__ import annotations

from pathlib import Path

import pytest

from finetree_annotator.deploy.runpod_endpoint import build_vllm_endpoint_env, write_env_file
from finetree_annotator.finetune.config import FinetuneConfig


def test_build_vllm_endpoint_env_uses_inference_model_override() -> None:
    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B", "max_seq_length": 1024},
            "inference": {"model_path": "asafd60/qwen35-merged", "adapter_path": "asafd60/qwen35-test"},
        }
    )
    env = build_vllm_endpoint_env(cfg, served_model_name="qwenasaf", gpu_memory_utilization=0.95)

    assert env["MODEL_NAME"] == "asafd60/qwen35-merged"
    assert env["OPENAI_SERVED_MODEL_NAME_OVERRIDE"] == "qwenasaf"
    assert env["MAX_MODEL_LEN"] == "1024"
    assert env["GPU_MEMORY_UTILIZATION"] == "0.95"
    assert env["FINETREE_ADAPTER_REF"] == "asafd60/qwen35-test"
    assert env["FINETREE_MODEL_FAMILY"] == "qwen3_5_a3"


def test_build_vllm_endpoint_env_rejects_bad_gpu_memory_utilization() -> None:
    cfg = FinetuneConfig.model_validate({})
    with pytest.raises(ValueError):
        build_vllm_endpoint_env(cfg, gpu_memory_utilization=0.0)


def test_write_env_file_writes_sorted_key_value_lines(tmp_path: Path) -> None:
    path = write_env_file({"B": "2", "A": "1"}, tmp_path / "runpod" / "endpoint.env")
    text = path.read_text(encoding="utf-8")
    assert text == "A=1\nB=2\n"
