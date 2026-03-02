from __future__ import annotations

import pytest

from finetree_annotator.finetune.config import FinetuneConfig, resolve_hf_token


def test_finetune_config_defaults_validate() -> None:
    cfg = FinetuneConfig.model_validate({})
    assert cfg.model.base_model
    assert cfg.data.bbox_policy == "include_if_present"
    assert cfg.data.include_empty_pages is True
    assert cfg.prompt.use_custom_prompt is True
    assert cfg.training.ddp_find_unused_parameters is False
    assert cfg.training.require_val_set is True
    assert cfg.training.compute_token_accuracy is True
    assert cfg.inference.quantization_mode == "none"
    assert cfg.inference.attn_implementation == "sdpa"
    assert cfg.inference.require_flash_attention is False
    assert cfg.inference.temperature == 0.7
    assert cfg.inference.top_p == 0.8
    assert cfg.inference.do_sample is True
    assert cfg.inference.enable_thinking is False
    assert cfg.inference.fallback_disable_adapter is True
    assert cfg.inference.max_memory_per_gpu_gb is None
    assert cfg.inference.gpu_memory_utilization == 0.9


def test_inference_backend_accepts_runpod_queue() -> None:
    cfg = FinetuneConfig.model_validate({"inference": {"backend": "runpod_queue"}})
    assert cfg.inference.backend == "runpod_queue"


def test_push_to_hub_requires_repo_and_token_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("FINETREE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(Exception):
        FinetuneConfig.model_validate({"push_to_hub": {"enabled": True, "repo_id": "abc"}})

    cfg = FinetuneConfig.model_validate(
        {
            "push_to_hub": {
                "enabled": True,
                "repo_id": "org/model",
                "hf_token": "hf_test",
            }
        }
    )
    assert cfg.push_to_hub.enabled is True
    assert cfg.push_to_hub.repo_id == "org/model"
    assert resolve_hf_token(cfg.push_to_hub) == "hf_test"


def test_push_to_hub_accepts_token_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FINETREE_HF_TOKEN", "hf_from_env")
    cfg = FinetuneConfig.model_validate(
        {
            "push_to_hub": {
                "enabled": True,
                "repo_id": "org/model",
                "hf_token": None,
            }
        }
    )
    assert cfg.push_to_hub.enabled is True
    assert resolve_hf_token(cfg.push_to_hub) == "hf_from_env"
