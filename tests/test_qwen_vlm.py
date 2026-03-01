from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from finetree_annotator import qwen_vlm
from finetree_annotator.finetune.config import FinetuneConfig


def test_resolve_config_path_uses_env(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("run: {}\n", encoding="utf-8")
    monkeypatch.setenv("FINETREE_QWEN_CONFIG", str(cfg))

    resolved = qwen_vlm._resolve_config_path(None)
    assert resolved == cfg.resolve()


def test_generate_page_extraction_parses_output(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"x")

    payload = (
        '{"meta":{"entity_name":null,"page_num":null,"type":"other","title":null},'
        '"facts":[{"bbox":{"x":1,"y":2,"w":3,"h":4},"value":"10",'
        '"refference":"","date":null,"path":[],"currency":null,"scale":null,"value_type":"amount"}]}'
    )

    monkeypatch.setattr(qwen_vlm, "generate_content_from_image", lambda **_: payload)
    extraction = qwen_vlm.generate_page_extraction_from_image(image_path=image_path, prompt="p")

    assert extraction.meta.type.value == "other"
    assert len(extraction.facts) == 1
    assert extraction.facts[0].bbox.x == 1


def test_generate_content_from_image_passes_max_new_tokens(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_stream_content_from_image(**kwargs):
        seen.update(kwargs)
        return iter(["ok"])

    monkeypatch.setattr(qwen_vlm, "stream_content_from_image", _fake_stream_content_from_image)
    output = qwen_vlm.generate_content_from_image(
        image_path=Path("/tmp/fake.png"),
        prompt="hello",
        max_new_tokens=7,
    )

    assert output == "ok"
    assert seen["max_new_tokens"] == 7


def test_resolve_adapter_reference_supports_hub_repo() -> None:
    adapter_ref, is_local = qwen_vlm._resolve_adapter_reference("asafd60/qwen35-test")
    assert adapter_ref == "asafd60/qwen35-test"
    assert is_local is False


def test_load_model_bundle_accepts_remote_adapter(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)

    calls: dict[str, object] = {}

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="flash_attention_2")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["model_name"] = model_name
            calls["model_kwargs"] = kwargs
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["processor_name"] = model_name
            calls["processor_kwargs"] = kwargs
            return object()

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model_obj, adapter_ref: str, **kwargs):
            calls["adapter_ref"] = adapter_ref
            calls["adapter_kwargs"] = kwargs
            return model_obj

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = _FakePeftModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "adapter_path": "asafd60/qwen35-test",
                "quantization_mode": "bnb_8bit",
                "attn_implementation": "flash_attention_2",
                "require_flash_attention": True,
            },
        }
    )
    qwen_vlm._load_model_bundle(cfg)

    assert calls["model_name"] == "unsloth/Qwen3.5-35B-A3B"
    assert calls["processor_name"] == "unsloth/Qwen3.5-35B-A3B"
    assert calls["adapter_ref"] == "asafd60/qwen35-test"
    assert calls["model_kwargs"]["load_in_8bit"] is True
    assert calls["model_kwargs"]["attn_implementation"] == "flash_attention_2"


def test_load_model_bundle_uses_4bit_alias(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)

    calls: dict[str, object] = {}

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="flash_attention_2")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["model_name"] = model_name
            calls["model_kwargs"] = kwargs
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {"load_in_4bit": True, "quantization_mode": "bnb_8bit"},
        }
    )
    qwen_vlm._load_model_bundle(cfg)

    assert calls["model_name"] == "unsloth/Qwen3.5-35B-A3B"
    assert calls["model_kwargs"]["load_in_4bit"] is True
    assert "load_in_8bit" not in calls["model_kwargs"]


def test_load_model_bundle_honors_env_model_adapter_and_8bit(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)
    monkeypatch.setenv("FINETREE_QWEN_MODEL", "org/custom-model")
    monkeypatch.setenv("FINETREE_ADAPTER_REF", "org/custom-adapter")
    monkeypatch.setenv("FINETREE_QWEN_QUANTIZATION", "bnb_8bit")

    calls: dict[str, object] = {}

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="flash_attention_2")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["model_name"] = model_name
            calls["model_kwargs"] = kwargs
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["processor_name"] = model_name
            return object()

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model_obj, adapter_ref: str, **kwargs):
            calls["adapter_ref"] = adapter_ref
            return model_obj

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = _FakePeftModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "adapter_path": None,
                "quantization_mode": "none",
                "attn_implementation": "flash_attention_2",
                "require_flash_attention": True,
            },
        }
    )
    qwen_vlm._load_model_bundle(cfg)

    assert calls["model_name"] == "org/custom-model"
    assert calls["processor_name"] == "org/custom-model"
    assert calls["adapter_ref"] == "org/custom-adapter"
    assert calls["model_kwargs"]["load_in_8bit"] is True


def test_load_model_bundle_requires_flash_attention(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="sdpa")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {"require_flash_attention": True, "attn_implementation": "flash_attention_2"},
        }
    )
    try:
        qwen_vlm._load_model_bundle(cfg)
    except RuntimeError as exc:
        assert "Flash attention is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected flash attention requirement failure")


def test_stream_content_from_image_uses_runpod_endpoint_backend(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "backend": "runpod_openai",
                "endpoint_base_url": "https://api.runpod.ai/v2/abc/openai/v1",
                "endpoint_api_key": "rp_test",
                "endpoint_model": "qwenasaf",
            },
        }
    )

    monkeypatch.setattr(qwen_vlm, "_resolve_config_path", lambda _cfg: tmp_path / "cfg.yaml")
    monkeypatch.setattr(qwen_vlm, "load_finetune_config", lambda _path: cfg)

    seen: dict[str, object] = {}

    class _Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"hello "}}]}',
                    'data: {"choices":[{"delta":{"content":"world"}}]}',
                    "data: [DONE]",
                ]
            )

        def read(self):
            return b""

    class _Client:
        def __init__(self, timeout):
            seen["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, endpoint, headers=None, json=None):
            seen["method"] = method
            seen["endpoint"] = endpoint
            seen["headers"] = headers
            seen["json"] = json
            return _Response()

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = _Client
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    output = "".join(
        qwen_vlm.stream_content_from_image(
            image_path=image_path,
            prompt="test prompt",
            model=None,
            config_path=str(tmp_path / "cfg.yaml"),
        )
    )
    assert output == "hello world"
    assert seen["endpoint"] == "https://api.runpod.ai/v2/abc/openai/v1/chat/completions"
    payload = seen["json"]  # type: ignore[assignment]
    assert payload["max_tokens"] == 120
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_stream_content_from_image_runpod_endpoint_adds_sampling_fields_when_enabled(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "backend": "runpod_openai",
                "endpoint_base_url": "https://api.runpod.ai/v2/abc/openai/v1",
                "endpoint_api_key": "rp_test",
                "endpoint_model": "qwenasaf",
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.8,
            },
        }
    )

    monkeypatch.setattr(qwen_vlm, "_resolve_config_path", lambda _cfg: tmp_path / "cfg.yaml")
    monkeypatch.setattr(qwen_vlm, "load_finetune_config", lambda _path: cfg)

    seen: dict[str, object] = {}

    class _Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_lines(self):
            return iter(["data: [DONE]"])

        def read(self):
            return b""

    class _Client:
        def __init__(self, timeout):
            seen["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, endpoint, headers=None, json=None):
            seen["json"] = json
            return _Response()

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = _Client
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    output = "".join(
        qwen_vlm.stream_content_from_image(
            image_path=image_path,
            prompt="test prompt",
            model=None,
            config_path=str(tmp_path / "cfg.yaml"),
        )
    )
    assert output == ""
    payload = seen["json"]  # type: ignore[assignment]
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8


def test_stream_content_from_image_uses_runpod_queue_backend(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "backend": "runpod_queue",
                "endpoint_base_url": "https://api.runpod.ai/v2/queue123",
                "endpoint_api_key": "rp_test",
                "endpoint_model": "qwenasaf",
                "endpoint_timeout_sec": 30,
            },
        }
    )

    monkeypatch.setattr(qwen_vlm, "_resolve_config_path", lambda _cfg: tmp_path / "cfg.yaml")
    monkeypatch.setattr(qwen_vlm, "load_finetune_config", lambda _path: cfg)
    monkeypatch.setattr(qwen_vlm.time, "sleep", lambda _seconds: None)

    seen: dict[str, object] = {}

    class _Response:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

        def read(self):
            return self.text.encode("utf-8")

    class _Client:
        def __init__(self, timeout):
            seen["timeout"] = timeout
            self._poll_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            seen["run_url"] = url
            seen["run_headers"] = headers
            seen["run_json"] = json
            return _Response({"id": "job-1", "status": "IN_QUEUE"})

        def get(self, url, headers=None):
            seen.setdefault("status_urls", []).append(url)
            seen["status_headers"] = headers
            self._poll_count += 1
            if self._poll_count == 1:
                return _Response({"id": "job-1", "status": "IN_PROGRESS"})
            return _Response({"id": "job-1", "status": "COMPLETED", "output": {"ok": True, "mode": "text", "text": "queue hello"}})

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = _Client
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    output = "".join(
        qwen_vlm.stream_content_from_image(
            image_path=image_path,
            prompt="test prompt",
            model=None,
            config_path=str(tmp_path / "cfg.yaml"),
            max_new_tokens=17,
        )
    )

    assert output == "queue hello"
    assert seen["run_url"] == "https://api.runpod.ai/v2/queue123/run"
    assert seen["status_urls"] == [
        "https://api.runpod.ai/v2/queue123/status/job-1",
        "https://api.runpod.ai/v2/queue123/status/job-1",
    ]
    assert seen["run_headers"] == {
        "Authorization": "Bearer rp_test",
        "Content-Type": "application/json",
    }
    payload_input = seen["run_json"]["input"]  # type: ignore[index]
    assert payload_input["response_mode"] == "text"
    assert payload_input["model"] == "qwenasaf"
    assert payload_input["max_tokens"] == 17
    assert isinstance(payload_input["image_base64"], str)


def test_stream_content_from_image_uses_runpod_queue_stream_endpoint(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "backend": "runpod_queue",
                "endpoint_base_url": "https://api.runpod.ai/v2/queue123",
                "endpoint_api_key": "rp_test",
                "endpoint_model": "qwenasaf",
                "endpoint_timeout_sec": 30,
            },
        }
    )

    monkeypatch.setattr(qwen_vlm, "_resolve_config_path", lambda _cfg: tmp_path / "cfg.yaml")
    monkeypatch.setattr(qwen_vlm, "load_finetune_config", lambda _path: cfg)

    seen: dict[str, object] = {}

    class _PostResponse:
        status_code = 200
        text = '{"id":"job-2","status":"IN_QUEUE"}'

        def json(self):
            return {"id": "job-2", "status": "IN_QUEUE"}

        def read(self):
            return self.text.encode("utf-8")

    class _StreamResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_lines(self):
            return iter(
                [
                    '[{"output":{"text":["hello "]}}]',
                    '[{"output":{"text":["world"]}}]',
                    "[DONE]",
                ]
            )

    class _Client:
        def __init__(self, timeout):
            seen["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            seen["run_url"] = url
            return _PostResponse()

        def stream(self, method, url, headers=None, timeout=None):
            seen["stream_method"] = method
            seen["stream_url"] = url
            seen["stream_timeout"] = timeout
            return _StreamResponse()

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = _Client
    fake_httpx.Timeout = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    output = "".join(
        qwen_vlm.stream_content_from_image(
            image_path=image_path,
            prompt="test prompt",
            model=None,
            config_path=str(tmp_path / "cfg.yaml"),
        )
    )

    assert output == "hello world"
    assert seen["run_url"] == "https://api.runpod.ai/v2/queue123/run"
    assert seen["stream_method"] == "GET"
    assert seen["stream_url"] == "https://api.runpod.ai/v2/queue123/stream/job-2"
    assert isinstance(seen["stream_timeout"], dict)
