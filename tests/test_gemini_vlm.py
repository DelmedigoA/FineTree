from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator import gemini_vlm


def _wrapper_payload(
    *,
    image: str = "page_0001.png",
    images_dir: str = "data/pdf_images/doc1",
    metadata: dict[str, object] | None = None,
    meta: dict[str, object] | None = None,
    facts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "images_dir": images_dir,
        "metadata": metadata or {},
        "pages": [
            {
                "image": image,
                "meta": meta
                or {
                    "entity_name": None,
                    "page_num": None,
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                },
                "facts": facts or [],
            }
        ],
    }


def test_resolve_api_key_prefers_google_api_key_env(monkeypatch) -> None:
    gemini_vlm._api_key_from_doppler.cache_clear()
    monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")
    monkeypatch.setenv("FINETREE_GEMINI_API_KEY", "finetree-env-key")
    assert gemini_vlm.resolve_api_key() == "google-env-key"


def test_resolve_api_key_supports_finetree_env_name(monkeypatch) -> None:
    gemini_vlm._api_key_from_doppler.cache_clear()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("FINETREE_GEMINI_API_KEY", "finetree-env-key")
    assert gemini_vlm.resolve_api_key() == "finetree-env-key"


def test_resolve_api_key_uses_doppler_when_env_missing(monkeypatch) -> None:
    gemini_vlm._api_key_from_doppler.cache_clear()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("FINETREE_GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_vlm, "_api_key_from_config", lambda: None)
    monkeypatch.setattr(gemini_vlm.shutil, "which", lambda _cmd: "/usr/local/bin/doppler")

    class _Proc:
        stdout = "doppler-key\n"

    def _fake_run(cmd, check, capture_output, text, timeout):
        assert cmd[:3] == ["doppler", "secrets", "get"]
        assert check is True
        assert capture_output is True
        assert text is True
        assert timeout == 5
        return _Proc()

    monkeypatch.setattr(gemini_vlm.subprocess, "run", _fake_run)
    assert gemini_vlm.resolve_api_key() == "doppler-key"


def test_build_generation_contents_legacy_shape_without_few_shot(tmp_path: Path, monkeypatch) -> None:
    target_image = tmp_path / "target.png"
    target_image.write_bytes(b"target")

    class _FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"kind": "bytes", "size": len(data), "mime": mime_type}

    class _FakeTypes:
        Part = _FakePart

    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)
    contents = gemini_vlm._build_generation_contents(
        image_path=target_image,
        prompt="extract now",
        mime_type=None,
        few_shot_examples=None,
    )

    assert len(contents) == 2
    assert contents[0]["kind"] == "bytes"
    assert contents[1] == "extract now"


def test_build_generation_contents_includes_few_shot_turns(tmp_path: Path, monkeypatch) -> None:
    target_image = tmp_path / "target.png"
    example_1 = tmp_path / "example_1.png"
    example_2 = tmp_path / "example_2.png"
    target_image.write_bytes(b"target")
    example_1.write_bytes(b"ex1")
    example_2.write_bytes(b"ex2")

    class _FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"kind": "bytes", "size": len(data), "mime": mime_type}

        @staticmethod
        def from_text(*, text):
            return {"kind": "text", "text": text}

    class _FakeTypes:
        Part = _FakePart

    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)
    contents = gemini_vlm._build_generation_contents(
        image_path=target_image,
        prompt="extract final",
        mime_type=None,
        few_shot_examples=[
            {"image_path": example_1, "expected_json": json.dumps(_wrapper_payload())},
            {
                "image_path": example_2,
                "expected_json": json.dumps(
                    _wrapper_payload(
                        meta={
                            "entity_name": None,
                            "page_num": None,
                            "page_type": "other",
                            "statement_type": None,
                            "title": None,
                        }
                    )
                ),
            },
        ],
    )

    assert len(contents) == 5
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[2]["role"] == "user"
    assert contents[3]["role"] == "model"
    assert contents[4]["role"] == "user"
    assert contents[4]["parts"][1]["kind"] == "text"
    assert contents[4]["parts"][1]["text"] == "extract final"


def test_generation_config_uses_high_thinking_for_gemini_3(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    cfg = gemini_vlm._generation_config("gemini-3-flash-preview", enable_thinking=True)

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_level": "HIGH"}


def test_generation_config_uses_minimal_for_gemini_3_nonthinking(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    cfg = gemini_vlm._generation_config("gemini-3-flash-preview", enable_thinking=False)

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_level": "MINIMAL"}


def test_generation_config_uses_zero_budget_for_nonthinking_legacy_models(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    cfg = gemini_vlm._generation_config("gemini-2.5-flash", enable_thinking=False)

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_budget": 0}


def test_parse_llm_json_accepts_quad_backtick_fence() -> None:
    payload = (
        "````json\n"
        + json.dumps(
            _wrapper_payload(
                meta={
                    "entity_name": None,
                    "page_num": "3",
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                },
                facts=[{"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "value": "10", "note_ref": None, "path": []}],
            )
        )
        + "\n"
        "````"
    )
    parsed = gemini_vlm._parse_llm_json(payload)
    assert isinstance(parsed, dict)
    assert isinstance(parsed.get("pages"), list)
    assert parsed["pages"][0]["facts"][0]["bbox"]["x"] == 1


def test_parse_llm_json_accepts_bbox_array_shape() -> None:
    payload = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "other",
                "statement_type": None,
                "title": None,
            },
            facts=[{"bbox": [1, 2, 3, 4], "value": "10", "note_ref": None, "path": []}],
        )
    )
    parsed = gemini_vlm._parse_llm_json(payload)
    assert isinstance(parsed, dict)
    assert isinstance(parsed.get("pages"), list)
    assert parsed["pages"][0]["facts"][0]["bbox"] == [1, 2, 3, 4]


def test_streaming_parser_finalize_falls_back_to_streamed_meta_and_facts() -> None:
    parser = gemini_vlm.StreamingPageExtractionParser()
    chunk = "```json\n" + json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "other",
                "statement_type": None,
                "title": None,
            },
            facts=[{"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "value": "10", "note_ref": None, "path": []}],
        )
    )
    meta, facts = parser.feed(chunk)
    assert meta is not None
    assert len(facts) == 1

    extraction = parser.finalize()
    assert extraction.meta.page_num == "3"
    assert len(extraction.facts) == 1
    assert extraction.facts[0].bbox.x == 1
