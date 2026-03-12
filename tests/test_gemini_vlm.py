from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_generate_content_from_image_writes_timestamped_log_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    example_image = tmp_path / "example.png"
    target_image.write_bytes(b"target")
    example_image.write_bytes(b"example")

    class _FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"kind": "bytes", "size": len(data), "mime": mime_type}

    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        LOW = "LOW"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        Part = _FakePart
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    class _FakeResponse:
        text = '{"ok":true}'

        def model_dump(self, mode="json"):
            _ = mode
            return {"text": self.text, "thinking": "hidden-thought"}

    class _FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-3.1-pro-preview"
            return _FakeResponse()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = _FakeModels()

    class _FakeGenAI:
        Client = _FakeClient

    monkeypatch.setattr(gemini_vlm, "genai", _FakeGenAI)
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    text = gemini_vlm.generate_content_from_image(
        image_path=target_image,
        prompt="extract now",
        model="Gemini Pro",
        api_key="k",
        few_shot_examples=[{"image_path": example_image, "expected_json": '{"pages":[]}'}],
        enable_thinking=True,
        thinking_level="high",
    )

    assert text == '{"ok":true}'
    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    request_payload = json.loads((log_dir / "request.json").read_text(encoding="utf-8"))
    response_payload = json.loads((log_dir / "response.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert request_payload["model"] == "gemini-3.1-pro-preview"
    assert request_payload["prompt"] == "extract now"
    assert request_payload["request_summary"]["model_family"] == "gemini_3"
    assert request_payload["request_summary"]["thinking_semantics"] == "thinking_level"
    assert request_payload["request_summary"]["effective_thinking_value"] == "HIGH"
    assert request_payload["exact_request"]["contents"][0]["role"] == "user"
    assert request_payload["exact_request"]["contents"][0]["parts"][0] == {"type": "image_file", "file": "few_shot_01_example.png"}
    assert request_payload["exact_request"]["contents"][-1]["parts"][0] == {"type": "image_file", "file": "input_target.png"}
    assert request_payload["few_shot_examples"][0]["expected_json"] == '{"pages":[]}'
    assert (log_dir / request_payload["logged_image_path"]).read_bytes() == b"target"
    assert (log_dir / request_payload["few_shot_examples"][0]["logged_image_path"]).read_bytes() == b"example"
    assert (log_dir / "output.txt").read_text(encoding="utf-8") == '{"ok":true}'
    assert response_payload["thinking"] == "hidden-thought"
    assert any(event["event"] == "thinking_request_summary" for event in events)
    thinking_response = next(event for event in events if event["event"] == "thinking_response_summary")
    assert thinking_response["observed_thinking_signal"] is True


def test_stream_content_from_image_writes_chunk_logs_and_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    target_image.write_bytes(b"target")

    class _FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"kind": "bytes", "size": len(data), "mime": mime_type}

    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        LOW = "LOW"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        Part = _FakePart
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    class _FakeChunk:
        def __init__(self, text: str, thought: str):
            self.text = text
            self.thought = thought

        def model_dump(self, mode="json"):
            _ = mode
            return {"text": self.text, "thought": self.thought}

    class _FakeModels:
        def generate_content_stream(self, **kwargs):
            assert kwargs["model"] == "gemini-3.1-pro-preview"
            return iter([
                _FakeChunk("part-1 ", "think-1"),
                _FakeChunk("part-2", "think-2"),
            ])

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = _FakeModels()

    class _FakeGenAI:
        Client = _FakeClient

    monkeypatch.setattr(gemini_vlm, "genai", _FakeGenAI)
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    chunks = list(
        gemini_vlm.stream_content_from_image(
            image_path=target_image,
            prompt="extract stream",
            model="Gemini Pro",
            api_key="k",
            enable_thinking=True,
            thinking_level="high",
        )
    )

    assert chunks == ["part-1 ", "part-2"]
    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    event_lines = (log_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    event_payloads = [json.loads(line) for line in event_lines]
    chunk_events = [payload for payload in event_payloads if payload["event"] == "stream_chunk"]
    request_payload = json.loads((log_dir / "request.json").read_text(encoding="utf-8"))

    assert len(chunk_events) == 2
    assert request_payload["request_summary"]["thinking_semantics"] == "thinking_level"
    assert request_payload["request_summary"]["effective_thinking_value"] == "HIGH"
    assert request_payload["exact_request"]["contents"] == [{"type": "image_file", "file": "input_target.png"}, "extract stream"]
    assert chunk_events[0]["raw"]["thought"] == "think-1"
    assert chunk_events[1]["raw"]["thought"] == "think-2"
    assert chunk_events[0]["observed_thinking_signal"] is True
    thinking_summary = next(payload for payload in event_payloads if payload["event"] == "thinking_response_summary")
    assert thinking_summary["observed_thinking_signal"] is True
    assert (log_dir / "output.txt").read_text(encoding="utf-8") == "part-1 part-2"


def test_stream_content_from_image_writes_partial_output_when_consumer_closes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    target_image.write_bytes(b"target")

    class _FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"kind": "bytes", "size": len(data), "mime": mime_type}

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeTypes:
        Part = _FakePart
        GenerateContentConfig = _FakeGenerateContentConfig

    class _FakeChunk:
        def __init__(self, text: str):
            self.text = text

        def model_dump(self, mode="json"):
            _ = mode
            return {"text": self.text}

    class _FakeModels:
        def generate_content_stream(self, **_kwargs):
            return iter([_FakeChunk("part-1 "), _FakeChunk("part-2")])

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = _FakeModels()

    class _FakeGenAI:
        Client = _FakeClient

    monkeypatch.setattr(gemini_vlm, "genai", _FakeGenAI)
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    stream = gemini_vlm.stream_content_from_image(
        image_path=target_image,
        prompt="extract stream",
        model="Gemini Pro",
        api_key="k",
    )
    assert next(stream) == "part-1 "
    stream.close()

    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    event_payloads = [
        json.loads(line)
        for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    aborted_events = [payload for payload in event_payloads if payload["event"] == "stream_aborted"]

    assert (log_dir / "output.txt").read_text(encoding="utf-8") == "part-1 "
    response_payload = json.loads((log_dir / "response.json").read_text(encoding="utf-8"))
    assert response_payload["status"] == "aborted"
    assert aborted_events


def test_stream_content_from_image_writes_partial_output_when_stream_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    target_image.write_bytes(b"target")

    class _FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"kind": "bytes", "size": len(data), "mime": mime_type}

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeTypes:
        Part = _FakePart
        GenerateContentConfig = _FakeGenerateContentConfig

    class _FakeChunk:
        def __init__(self, text: str):
            self.text = text

        def model_dump(self, mode="json"):
            _ = mode
            return {"text": self.text}

    class _FakeModels:
        def generate_content_stream(self, **_kwargs):
            def _gen():
                yield _FakeChunk("part-1 ")
                raise RuntimeError("stream boom")

            return _gen()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = _FakeModels()

    class _FakeGenAI:
        Client = _FakeClient

    monkeypatch.setattr(gemini_vlm, "genai", _FakeGenAI)
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    with pytest.raises(RuntimeError, match="stream boom"):
        list(
            gemini_vlm.stream_content_from_image(
                image_path=target_image,
                prompt="extract stream",
                model="Gemini Pro",
                api_key="k",
            )
        )

    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    event_payloads = [
        json.loads(line)
        for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    error_events = [payload for payload in event_payloads if payload["event"] == "stream_error"]

    assert (log_dir / "output.txt").read_text(encoding="utf-8") == "part-1 "
    response_payload = json.loads((log_dir / "response.json").read_text(encoding="utf-8"))
    assert response_payload["status"] == "error"
    assert response_payload["error"] == "stream boom"
    assert error_events


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


def test_generation_config_uses_low_for_gemini_3_pro_nonthinking(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    cfg = gemini_vlm._generation_config("gemini-3.1-pro-preview", enable_thinking=False)

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_level": "LOW"}


def test_generation_config_supports_explicit_medium_level(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    cfg = gemini_vlm._generation_config("gemini-3.1-flash-lite", thinking_level="medium")

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_level": "MEDIUM"}


def test_generation_config_coerces_minimal_to_low_for_gemini_3_pro(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    cfg = gemini_vlm._generation_config("gemini-3.1-pro-preview", thinking_level="minimal")

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_level": "LOW"}


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


def test_generation_config_uses_auto_budget_for_thinking_legacy_models(monkeypatch) -> None:
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

    cfg = gemini_vlm._generation_config("gemini-2.5-flash", enable_thinking=True)

    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["thinking_config"].kwargs == {"thinking_budget": -1}


def test_generation_config_rejects_unknown_thinking_level(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeThinkingLevel:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
        MINIMAL = "MINIMAL"

    class _FakeTypes:
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel

    monkeypatch.setattr(gemini_vlm, "genai", object())
    monkeypatch.setattr(gemini_vlm, "types", _FakeTypes)

    with pytest.raises(ValueError, match="thinking_level"):
        gemini_vlm._generation_config("gemini-3.1-flash-lite", thinking_level="ultra")


def test_resolve_supported_gemini_model_name_maps_generic_pro_alias() -> None:
    assert gemini_vlm.resolve_supported_gemini_model_name("Gemini Pro") == "gemini-3.1-pro-preview"


def test_resolve_supported_gemini_model_name_preserves_supported_exact_model() -> None:
    assert gemini_vlm.resolve_supported_gemini_model_name("gemini-2.5-pro") == "gemini-2.5-pro"


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


def test_parse_selected_field_patch_text_valid_payload() -> None:
    payload = {
        "meta_updates": {"statement_type": "income_statement"},
        "fact_updates": [
            {
                "fact_num": 2,
                "updates": {
                    "row_role": "total",
                    "period_type": "duration",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                    "path_source": "observed",
                },
            }
        ],
    }
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(payload),
        allowed_fact_fields={"row_role", "period_type", "period_start", "period_end", "path_source"},
        allow_statement_type=True,
    )
    assert parsed["meta_updates"]["statement_type"] == "income_statement"
    assert parsed["fact_updates"][0]["fact_num"] == 2
    assert parsed["fact_updates"][0]["updates"]["row_role"] == "total"
    assert parsed["fact_updates"][0]["updates"]["period_type"] == "duration"


def test_parse_selected_field_patch_text_accepts_canonical_equations_updates() -> None:
    payload = {
        "meta_updates": {},
        "fact_updates": [
            {
                "fact_num": 3,
                "updates": {
                    "row_role": "total",
                    "equations": [
                        {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                        {"equation": "120", "fact_equation": None},
                    ],
                },
            }
        ],
    }
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(payload),
        allowed_fact_fields={"row_role", "equations"},
        allow_statement_type=False,
    )

    assert parsed["fact_updates"] == [
        {
            "fact_num": 3,
            "updates": {
                "row_role": "total",
                "equations": [
                    {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                    {"equation": "120", "fact_equation": None},
                ],
            },
        }
    ]


def test_parse_selected_field_patch_text_accepts_legacy_equation_keys_when_equations_requested() -> None:
    payload = {
        "meta_updates": {},
        "fact_updates": [
            {
                "fact_num": 3,
                "updates": {
                    "row_role": "total",
                    "equation": "100 + 20",
                    "fact_equation": "f1 + f2",
                },
            }
        ],
    }
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(payload),
        allowed_fact_fields={"row_role", "equations"},
        allow_statement_type=False,
    )

    assert parsed["fact_updates"] == [
        {
            "fact_num": 3,
            "updates": {
                "row_role": "total",
                "equations": [
                    {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                ],
            },
        }
    ]


def test_parse_selected_field_patch_text_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValueError, match="unknown top-level keys"):
        gemini_vlm.parse_selected_field_patch_text(
            json.dumps(
                {
                    "meta_updates": {},
                    "fact_updates": [],
                    "unexpected": True,
                }
            ),
            allowed_fact_fields={"period_type"},
            allow_statement_type=False,
        )


def test_parse_selected_field_patch_text_ignores_unrequested_statement_type() -> None:
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(
            {
                "meta_updates": {"statement_type": "income_statement"},
                "fact_updates": [{"fact_num": 1, "updates": {"period_type": "instant"}}],
            }
        ),
        allowed_fact_fields={"period_type"},
        allow_statement_type=False,
    )

    assert parsed["meta_updates"] == {}
    assert parsed["fact_updates"] == [{"fact_num": 1, "updates": {"period_type": "instant"}}]


def test_parse_selected_field_patch_text_rejects_duplicate_or_invalid_fact_num() -> None:
    with pytest.raises(ValueError, match="Duplicate fact_num"):
        gemini_vlm.parse_selected_field_patch_text(
            json.dumps(
                {
                    "meta_updates": {},
                    "fact_updates": [
                        {"fact_num": 1, "updates": {"period_type": "instant"}},
                        {"fact_num": 1, "updates": {"period_type": "duration"}},
                    ],
                }
            ),
            allowed_fact_fields={"period_type"},
            allow_statement_type=False,
        )

    with pytest.raises(ValueError, match="fact_num must be an integer"):
        gemini_vlm.parse_selected_field_patch_text(
            json.dumps(
                {
                    "meta_updates": {},
                    "fact_updates": [
                        {"fact_num": "1", "updates": {"period_type": "instant"}},
                    ],
                }
            ),
            allowed_fact_fields={"period_type"},
            allow_statement_type=False,
        )


def test_parse_selected_field_patch_text_rejects_non_requested_update_key() -> None:
    with pytest.raises(ValueError, match="non-requested keys"):
        gemini_vlm.parse_selected_field_patch_text(
            json.dumps(
                {
                    "meta_updates": {},
                    "fact_updates": [
                        {"fact_num": 1, "updates": {"period_type": "instant", "note_ref": "n1"}},
                    ],
                }
            ),
            allowed_fact_fields={"period_type"},
            allow_statement_type=False,
        )
