from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from finetree_annotator.finetune.config import load_finetune_config, resolve_hf_token
from finetree_annotator.finetune.dataset_builder import build_unsloth_chat_datasets


_RUN_GPU_TESTS = os.getenv("FINETREE_RUN_GPU_TESTS") == "1"
_RUN_HF_TESTS = os.getenv("FINETREE_RUN_HF_TESTS") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _RUN_GPU_TESTS, reason="Set FINETREE_RUN_GPU_TESTS=1 to run GPU tests."),
]


def test_cuda_visible_and_working() -> None:
    torch = importlib.import_module("torch")
    assert torch.cuda.is_available(), "CUDA not available in this VM."
    device_count = torch.cuda.device_count()
    assert device_count >= 1, "No CUDA devices found."

    # Basic kernel execution smoke test.
    x = torch.randn((1024, 1024), device="cuda")
    y = torch.randn((1024, 1024), device="cuda")
    z = x @ y
    assert z.is_cuda
    assert z.shape == (1024, 1024)


def test_gpu_software_stack_imports() -> None:
    modules = [
        "transformers",
        "datasets",
        "trl",
        "peft",
        "huggingface_hub",
        "unsloth",
    ]
    missing: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    assert not missing, f"Missing training stack modules: {missing}"


def test_runpod_config_and_dataset_build_smoke(tmp_path: Path) -> None:
    cfg = load_finetune_config(Path("configs/finetune_qwen35a3_vl.yaml"))
    cfg.data.output_train_jsonl = tmp_path / "train.jsonl"
    cfg.data.output_val_jsonl = tmp_path / "val.jsonl"

    stats = build_unsloth_chat_datasets(cfg)
    assert stats.annotation_files >= 1
    assert stats.pages_seen >= 1
    assert stats.samples_written_train + stats.samples_written_val >= 1
    assert cfg.model.base_model


@pytest.mark.hf
@pytest.mark.skipif(not _RUN_HF_TESTS, reason="Set FINETREE_RUN_HF_TESTS=1 to run Hugging Face connectivity test.")
def test_hf_connection_with_configured_token() -> None:
    cfg = load_finetune_config(Path("configs/finetune_qwen35a3_vl.yaml"))
    token = resolve_hf_token(cfg.push_to_hub)
    assert token, "HF token is missing. Set FINETREE_HF_TOKEN or HF_TOKEN."
    assert str(token).startswith("hf_"), "HF token does not look valid (must start with hf_)."

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    identity = api.whoami()
    assert isinstance(identity, dict), "HF whoami response is invalid."
