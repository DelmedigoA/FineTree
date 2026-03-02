from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from finetree_annotator import qwen_vlm
from finetree_annotator.finetune.config import FinetuneConfig


@pytest.fixture(autouse=True)
def _clear_qwen_env(monkeypatch) -> None:
    for name in (
        "FINETREE_QWEN_MODEL",
        "FINETREE_ADAPTER_REF",
        "FINETREE_QWEN_ADAPTER_PATH",
        "FINETREE_QWEN_ALLOW_ENV_ADAPTER_OVERRIDE",
        "FINETREE_QWEN_QUANTIZATION",
        "FINETREE_QWEN_LOAD_IN_4BIT",
        "FINETREE_QWEN_FALLBACK_MODEL",
        "FINETREE_QWEN_MAX_MEMORY_PER_GPU_GB",
        "FINETREE_QWEN_GPU_MEMORY_UTILIZATION",
        "FINETREE_QWEN_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)


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
        '"note":"*without debt insurance","is_beur":true,"beur_num":"5","refference":"","date":null,'
        '"path":[],"currency":null,"scale":null,"value_type":"amount"}]}'
    )

    monkeypatch.setattr(qwen_vlm, "generate_content_from_image", lambda **_: payload)
    extraction = qwen_vlm.generate_page_extraction_from_image(image_path=image_path, prompt="p")

    assert extraction.meta.type.value == "other"
    assert len(extraction.facts) == 1
    assert extraction.facts[0].bbox.x == 1
    assert extraction.facts[0].note == "*without debt insurance"
    assert extraction.facts[0].is_beur is True
    assert extraction.facts[0].beur_num == "5"


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


def test_apply_chat_template_with_thinking_control_prefers_enable_thinking_arg() -> None:
    seen: dict[str, object] = {}

    class _Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return "chat"

    result = qwen_vlm._apply_chat_template_with_thinking_control(
        processor=_Processor(),
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
    )
    assert result == "chat"
    kwargs = seen["kwargs"]  # type: ignore[assignment]
    assert kwargs["enable_thinking"] is False
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True


def test_apply_chat_template_with_thinking_control_falls_back_to_chat_template_kwargs() -> None:
    seen: dict[str, object] = {}

    class _Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return "chat"

    result = qwen_vlm._apply_chat_template_with_thinking_control(
        processor=_Processor(),
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
    )
    assert result == "chat"
    kwargs = seen["kwargs"]  # type: ignore[assignment]
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}


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

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model_obj, adapter_ref: str, **kwargs):
            calls["adapter_ref"] = adapter_ref
            calls["adapter_kwargs"] = kwargs
            return model_obj

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_transformers.BitsAndBytesConfig = _FakeBitsAndBytesConfig
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
    quant_cfg = calls["model_kwargs"]["quantization_config"]  # type: ignore[index]
    assert isinstance(quant_cfg, _FakeBitsAndBytesConfig)
    assert quant_cfg.kwargs["load_in_8bit"] is True
    assert "load_in_8bit" not in calls["model_kwargs"]
    assert calls["model_kwargs"]["attn_implementation"] == "flash_attention_2"


def test_load_model_bundle_uses_4bit_alias(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    sentinel_dtype = object()
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: sentinel_dtype)
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

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_transformers.BitsAndBytesConfig = _FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {"load_in_4bit": True, "quantization_mode": "bnb_8bit"},
        }
    )
    qwen_vlm._load_model_bundle(cfg)

    assert calls["model_name"] == "unsloth/Qwen3.5-35B-A3B"
    quant_cfg = calls["model_kwargs"]["quantization_config"]  # type: ignore[index]
    assert isinstance(quant_cfg, _FakeBitsAndBytesConfig)
    assert quant_cfg.kwargs["load_in_4bit"] is True
    assert quant_cfg.kwargs["bnb_4bit_compute_dtype"] is sentinel_dtype
    assert "load_in_4bit" not in calls["model_kwargs"]
    assert "load_in_8bit" not in calls["model_kwargs"]


def test_load_model_bundle_falls_back_to_sdpa_when_flash_attn_missing(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)

    calls: list[dict[str, object]] = []

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="sdpa")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls.append({"model_name": model_name, "kwargs": kwargs})
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise ImportError("flash_attn seems to be not installed")
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
            "inference": {
                "attn_implementation": "flash_attention_2",
                "require_flash_attention": False,
            },
        }
    )
    qwen_vlm._load_model_bundle(cfg)

    assert len(calls) == 2
    assert calls[0]["kwargs"]["attn_implementation"] == "flash_attention_2"  # type: ignore[index]
    assert calls[1]["kwargs"]["attn_implementation"] == "sdpa"  # type: ignore[index]


def test_load_model_bundle_honors_env_model_adapter_and_8bit(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)
    monkeypatch.setenv("FINETREE_QWEN_MODEL", "org/custom-model")
    monkeypatch.setenv("FINETREE_ADAPTER_REF", "org/custom-adapter")
    monkeypatch.setenv("FINETREE_QWEN_ALLOW_ENV_ADAPTER_OVERRIDE", "1")
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

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model_obj, adapter_ref: str, **kwargs):
            calls["adapter_ref"] = adapter_ref
            return model_obj

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_transformers.BitsAndBytesConfig = _FakeBitsAndBytesConfig
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
    quant_cfg = calls["model_kwargs"]["quantization_config"]  # type: ignore[index]
    assert isinstance(quant_cfg, _FakeBitsAndBytesConfig)
    assert quant_cfg.kwargs["load_in_8bit"] is True
    assert "load_in_8bit" not in calls["model_kwargs"]


def test_load_model_bundle_ignores_adapter_env_by_default(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)
    monkeypatch.setenv("FINETREE_ADAPTER_REF", "org/custom-adapter")

    calls: dict[str, object] = {}

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="flash_attention_2")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["model_name"] = model_name
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
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
                "attn_implementation": "flash_attention_2",
                "require_flash_attention": True,
            },
        }
    )
    qwen_vlm._load_model_bundle(cfg)
    assert "adapter_ref" not in calls


def test_build_max_memory_map_uses_gpu_memory_utilization(monkeypatch) -> None:
    class _Props:
        def __init__(self, total_memory: int):
            self.total_memory = total_memory

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 3,
        get_device_properties=lambda _idx: _Props(48 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {"gpu_memory_utilization": 0.9},
        }
    )
    max_memory = qwen_vlm._build_max_memory_map(cfg)
    assert max_memory == {0: "43GiB", 1: "43GiB", 2: "43GiB"}


def test_build_max_memory_map_honors_env_override(monkeypatch) -> None:
    class _Props:
        def __init__(self, total_memory: int):
            self.total_memory = total_memory

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda _idx: _Props(80 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("FINETREE_QWEN_MAX_MEMORY_PER_GPU_GB", "40")

    cfg = FinetuneConfig.model_validate(
        {
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {"gpu_memory_utilization": 0.95},
        }
    )
    max_memory = qwen_vlm._build_max_memory_map(cfg)
    assert max_memory == {0: "40GiB", 1: "40GiB"}


def test_load_model_bundle_falls_back_to_original_model_without_adapter(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)
    monkeypatch.setenv("FINETREE_QWEN_FALLBACK_MODEL", "Qwen/Qwen3.5-27B-Instruct")

    calls: dict[str, object] = {"model_names": []}

    class _FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(attn_implementation="sdpa")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["model_names"].append(model_name)  # type: ignore[union-attr]
            if model_name == "unsloth/Qwen3.5-35B-A3B":
                raise RuntimeError("primary failed")
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
                "adapter_path": "asafd60/qwen35-test",
                "fallback_disable_adapter": True,
            },
        }
    )

    with pytest.warns(RuntimeWarning, match="Falling back to original model"):
        qwen_vlm._load_model_bundle(cfg)

    assert calls["model_names"] == ["unsloth/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-27B-Instruct"]
    assert calls["processor_name"] == "Qwen/Qwen3.5-27B-Instruct"
    assert "adapter_ref" not in calls


def test_load_model_bundle_drops_adapter_on_cuda_oom_when_fallback_enabled(monkeypatch) -> None:
    qwen_vlm._MODEL_CACHE.clear()
    monkeypatch.setattr(qwen_vlm, "_ensure_cuda", lambda: None)
    monkeypatch.setattr(qwen_vlm, "_dtype_from_name", lambda _: None)
    monkeypatch.setattr(qwen_vlm, "resolve_hf_token_from_env", lambda: None)

    calls: dict[str, object] = {"adapter_calls": 0}

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

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model_obj, adapter_ref: str, **kwargs):
            calls["adapter_calls"] = int(calls["adapter_calls"]) + 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")

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
                "fallback_disable_adapter": True,
            },
        }
    )

    with pytest.warns(RuntimeWarning, match="Continuing without adapter"):
        qwen_vlm._load_model_bundle(cfg)

    assert calls["adapter_calls"] == 1


def test_load_model_bundle_raises_on_cuda_oom_when_fallback_disabled(monkeypatch) -> None:
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

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model_obj, adapter_ref: str, **kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")

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
                "fallback_disable_adapter": False,
            },
        }
    )

    with pytest.raises(RuntimeError, match="Adapter load failed due CUDA OOM"):
        qwen_vlm._load_model_bundle(cfg)


def test_load_model_bundle_falls_back_to_legacy_quant_kwargs_without_bitsandbytes(monkeypatch) -> None:
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
            "inference": {"quantization_mode": "bnb_8bit"},
        }
    )
    qwen_vlm._load_model_bundle(cfg)
    assert calls["model_kwargs"]["load_in_8bit"] is True


def test_load_with_optional_token_does_not_retry_on_unrelated_typeerror() -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def _load_fn(ref: str, **kwargs):
        seen.append((ref, kwargs))
        raise TypeError("unexpected keyword argument 'load_in_8bit'")

    try:
        qwen_vlm._load_with_optional_token(_load_fn, "org/model", "hf_123")
    except TypeError as exc:
        assert "load_in_8bit" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected TypeError")

    assert len(seen) == 1
    assert seen[0][1]["token"] == "hf_123"


def test_load_with_optional_token_retries_with_use_auth_token_on_token_typeerror() -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def _load_fn(ref: str, **kwargs):
        seen.append((ref, kwargs))
        if "token" in kwargs:
            raise TypeError("unexpected keyword argument 'token'")
        return "ok"

    result = qwen_vlm._load_with_optional_token(_load_fn, "org/model", "hf_123")
    assert result == "ok"
    assert len(seen) == 2
    assert seen[0][1]["token"] == "hf_123"
    assert seen[1][1]["use_auth_token"] == "hf_123"


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
                "max_new_tokens": 120,
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
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


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
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


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
    assert payload_input["chat_template_kwargs"] == {"enable_thinking": False}
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
