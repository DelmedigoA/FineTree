from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from finetree_annotator.deploy import simple_infer_api as mod


def _to_b64_png(width: int = 64, height: int = 96) -> str:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_resize_to_qwen_max_pixels_respects_factor_and_budget() -> None:
    img = Image.new("RGB", (1657, 2301), color=(255, 255, 255))
    out = mod.resize_to_qwen_max_pixels(img, max_pixels=1_200_000, factor=28)
    w, h = out.size
    assert w % 28 == 0
    assert h % 28 == 0
    assert w * h <= 1_200_000


def test_decode_image_rejects_invalid_base64() -> None:
    with pytest.raises(Exception):
        mod.decode_image("not_base64")


def test_load_runtime_config_reads_yaml_and_env_overrides(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "api.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model_id: local/model",
                "max_pixels: 222222",
                "temperature: 0.3",
                "do_sample: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("FINETREE_SIMPLE_API_MODEL_ID", "env/model")
    monkeypatch.setenv("FINETREE_SIMPLE_API_MAX_NEW_TOKENS", "777")
    cfg = mod.load_runtime_config(str(cfg_path))
    assert cfg.model_id == "env/model"
    assert cfg.max_pixels == 222222
    assert cfg.max_new_tokens == 777
    assert cfg.temperature == 0.3
    assert cfg.do_sample is True


def test_infer_endpoint_with_preloaded_runtime() -> None:
    class _FakeTokenizer:
        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            _ = skip_special_tokens
            return ",".join(str(int(x)) for x in tokens.tolist())

    class _FakeProcessor:
        def __init__(self):
            self.tokenizer = _FakeTokenizer()

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            _ = messages, kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def eval(self):
            return self

        @staticmethod
        def generate(**kwargs):
            _ = kwargs
            return torch.tensor([[1, 2, 3, 42, 43]], dtype=torch.long)

    app = mod.create_app(preloaded=(_FakeModel(), _FakeProcessor(), "cpu"), load_on_startup=False)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["loaded"] is True

    payload = {
        "image_b64": _to_b64_png(),
        "prompt": "extract",
        "max_new_tokens": 7,
    }
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    assert response.json()["text"] == "42,43"

