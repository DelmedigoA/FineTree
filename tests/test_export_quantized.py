from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from finetree_annotator.finetune.config import FinetuneConfig
from finetree_annotator.finetune import export_quantized


def test_export_quantized_model_defaults_to_8bit(tmp_path: Path, monkeypatch) -> None:
    cfg = FinetuneConfig.model_validate(
        {
            "run": {"output_dir": str(tmp_path / "artifacts")},
            "model": {"base_model": "unsloth/Qwen3.5-35B-A3B"},
            "inference": {
                "quantization_mode": "bnb_8bit",
                "attn_implementation": "flash_attention_2",
                "require_flash_attention": True,
            },
        }
    )

    seen: dict[str, object] = {}

    class _FakeModel:
        def save_pretrained(self, path: str, safe_serialization: bool = True):
            seen["saved_model_path"] = path
            seen["safe_serialization"] = safe_serialization

    class _FakeProcessor:
        def save_pretrained(self, path: str):
            seen["saved_processor_path"] = path

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(ref: str, **kwargs):
            seen.setdefault("model_calls", []).append((ref, kwargs))
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(ref: str, **kwargs):
            seen.setdefault("processor_calls", []).append((ref, kwargs))
            return _FakeProcessor()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _FakeAutoModel
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    out_dir = export_quantized.export_quantized_model(cfg, validate_reload=True)
    assert out_dir == tmp_path / "artifacts" / "merged-8bit"
    assert seen["safe_serialization"] is True
    first_call = seen["model_calls"][0]  # type: ignore[index]
    assert first_call[1]["load_in_8bit"] is True  # type: ignore[index]
    assert first_call[1]["attn_implementation"] == "flash_attention_2"  # type: ignore[index]

    manifest = json.loads((out_dir / "quantization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quantization_mode"] == "bnb_8bit"
    assert manifest["require_flash_attention"] is True


def test_effective_quantization_mode_honors_4bit_alias() -> None:
    cfg = FinetuneConfig.model_validate(
        {
            "inference": {
                "quantization_mode": "bnb_8bit",
                "load_in_4bit": True,
            }
        }
    )
    assert export_quantized._effective_quantization_mode(cfg, None) == "bnb_4bit"
