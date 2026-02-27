from __future__ import annotations

import pytest

from finetree_annotator.finetune.config import FinetuneConfig, resolve_hf_token


def test_finetune_config_defaults_validate() -> None:
    cfg = FinetuneConfig.model_validate({})
    assert cfg.model.base_model
    assert cfg.data.bbox_policy == "include_if_present"
    assert cfg.prompt.use_custom_prompt is True


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
