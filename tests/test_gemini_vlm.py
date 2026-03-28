from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import types
from typing import Any

import pytest
from PIL import Image

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


def _prompt_only_payload(
    *,
    image: str = "page_0001.png",
    meta: dict[str, object] | None = None,
    facts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
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
        ]
    }


def _create_logged_session(
    tmp_path: Path,
    *,
    image_size: tuple[int, int] = (800, 800),
    prompt: str | None = None,
) -> tuple[Path, Path]:
    image_path = tmp_path / "page.png"
    Image.new("RGB", image_size, color=(255, 255, 255)).save(image_path)
    session_dir = gemini_vlm._create_gemini_log_session(
        operation="stream_content",
        model="gemini-3-flash-preview",
        image_path=image_path,
        prompt=prompt or f"Current image size: {image_size[0]} x {image_size[1]} pixels",
        mime_type=None,
        system_prompt=None,
        temperature=0.01,
        enable_thinking=False,
        thinking_level=None,
        few_shot_examples=None,
    )
    return session_dir, image_path


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


def test_resolve_supported_gemini_model_name_preserves_vertex_tuned_model() -> None:
    assert gemini_vlm.resolve_supported_gemini_model_name("gemini-flash-hf-tuned") == "gemini-flash-hf-tuned"
    assert gemini_vlm.is_vertex_gemini_model_requested("Gemini Flash HF Tuned") is True


def test_resolve_vertex_project_id_defaults_to_tuned_project(monkeypatch) -> None:
    monkeypatch.delenv("FINETREE_VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setattr(gemini_vlm.shutil, "which", lambda _cmd: None)

    assert gemini_vlm.resolve_vertex_project_id() == "gen-lang-client-0533315636"


def test_resolve_vertex_endpoint_id_defaults_to_new_tuned_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("FINETREE_VERTEX_ENDPOINT_ID", raising=False)

    assert gemini_vlm._resolve_vertex_endpoint_id() == "4766539037060104192"


def test_resolve_vertex_region_defaults_to_europe_west4(monkeypatch) -> None:
    monkeypatch.delenv("FINETREE_VERTEX_REGION", raising=False)

    assert gemini_vlm._resolve_vertex_region() == "europe-west4"


def _install_fake_qwen_vl_utils(monkeypatch, *, height: int, width: int) -> None:
    vision_mod = types.ModuleType("qwen_vl_utils.vision_process")

    def _smart_resize(orig_height: int, orig_width: int, **kwargs):
        assert kwargs.get("max_pixels") == gemini_vlm.DEFAULT_VERTEX_TUNED_MAX_PIXELS
        return height, width

    vision_mod.smart_resize = _smart_resize
    root_mod = types.ModuleType("qwen_vl_utils")
    root_mod.vision_process = vision_mod
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", root_mod)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_mod)


def test_prepare_vertex_tuned_gemini_image_resizes_with_qwen_utils(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (2000, 1000), color=(255, 255, 255)).save(image_path)
    _install_fake_qwen_vl_utils(monkeypatch, height=756, width=1512)

    prepared = gemini_vlm._prepare_vertex_tuned_gemini_image(image_path, mime_type=None)

    assert prepared.original_width == 2000
    assert prepared.original_height == 1000
    assert prepared.prepared_width == 1512
    assert prepared.prepared_height == 756
    assert prepared.resized is True
    assert prepared.mime_type == "image/png"
    assert prepared.prepared_width * prepared.prepared_height <= gemini_vlm.DEFAULT_VERTEX_TUNED_MAX_PIXELS


def test_restore_resized_bbox_text_scales_pixel_boxes_back_to_original() -> None:
    prepared = gemini_vlm._PreparedGeminiImage(
        image_bytes=b"",
        mime_type="image/png",
        original_width=2000,
        original_height=1000,
        prepared_width=1000,
        prepared_height=500,
    )
    raw_text = json.dumps(
        _wrapper_payload(
            facts=[
                {"bbox": [100, 50, 20, 10], "value": "10", "path": []},
            ]
        ),
        ensure_ascii=False,
    )

    restored = gemini_vlm._restore_resized_bbox_text(raw_text, prepared)
    payload = json.loads(restored)

    assert payload["pages"][0]["facts"][0]["bbox"] == [200.0, 100.0, 40.0, 20.0]


def test_restore_resized_bbox_text_scales_even_when_bbox_values_fit_under_1000() -> None:
    prepared = gemini_vlm._PreparedGeminiImage(
        image_bytes=b"",
        mime_type="image/png",
        original_width=2000,
        original_height=1000,
        prepared_width=1000,
        prepared_height=500,
    )
    raw_text = json.dumps(
        _wrapper_payload(
            facts=[
                {"bbox": [500, 500, 200, 100], "value": "10", "path": []},
            ]
        ),
        ensure_ascii=False,
    )

    restored = gemini_vlm._restore_resized_bbox_text(raw_text, prepared)
    payload = json.loads(restored)

    assert payload["pages"][0]["facts"][0]["bbox"] == [1000.0, 1000.0, 400.0, 200.0]


def test_parse_bbox_only_text_accepts_simple_facts_shape() -> None:
    facts = gemini_vlm.parse_bbox_only_text(
        '{"pages":[{"image":"page_0001.png","meta":{"page_type":"other","statement_type":null},"facts":[{"bbox":[10,20,30,40],"value":"10"},{"bounding_box":[50,60,70,80],"value":"20"}]}]}'
    )

    assert facts == [
        {"bbox": [10.0, 20.0, 30.0, 40.0], "value": "10"},
        {"bbox": [50.0, 60.0, 70.0, 80.0], "value": "20"},
    ]


def test_streaming_bbox_only_parser_emits_complete_bbox_objects_before_finalize() -> None:
    parser = gemini_vlm.StreamingBBoxOnlyParser()

    meta_1, facts_1 = parser.feed('{"pages":[{"image":"page_0001.png","meta":{"page_type":"other","statement_type":null},"facts":[{"bbox":[10,20,30,40],"value":"10"},')
    meta_2, facts_2 = parser.feed('{"bounding_box":[50,60,70,80],"value":"20"}]}]}')

    assert meta_1 == {"entity_name": None, "page_num": None, "page_type": "other", "statement_type": None, "title": None}
    assert facts_1 == [{"bbox": [10.0, 20.0, 30.0, 40.0], "value": "10"}]
    assert meta_2 is None
    assert facts_2 == [{"bbox": [50.0, 60.0, 70.0, 80.0], "value": "20"}]
    assert parser.finalize().facts == [
        {"bbox": [10.0, 20.0, 30.0, 40.0], "value": "10"},
        {"bbox": [50.0, 60.0, 70.0, 80.0], "value": "20"},
    ]


def test_streaming_bbox_only_parser_salvages_valid_boxes_from_malformed_tail() -> None:
    parser = gemini_vlm.StreamingBBoxOnlyParser()

    parser.feed('{"pages":[{"image":"page_0001.png","meta":{"page_type":"other","statement_type":null},"facts":[{"bbox":[394,266,65,12],"value":"123"},{"bbox":[289,266,56,12],"value":"456"},')
    parser.feed('{"bbox":{"ymin_row":521,"xmin_col":218,"xmax_row":225')

    assert parser.finalize().facts == [
        {"bbox": [394.0, 266.0, 65.0, 12.0], "value": "123"},
        {"bbox": [289.0, 266.0, 56.0, 12.0], "value": "456"},
    ]


def test_vertex_tuned_effective_prompt_switches_bbox_instruction_to_resized_space() -> None:
    prepared = gemini_vlm._PreparedGeminiImage(
        image_bytes=b"",
        mime_type="image/png",
        original_width=1654,
        original_height=2339,
        prepared_width=896,
        prepared_height=1288,
    )
    prompt = "bbox must tightly cover the value text only, in pixel coordinates of the original image."

    effective = gemini_vlm._vertex_tuned_effective_prompt(prompt, prepared)

    assert "pixel coordinates of the original image" not in effective
    assert "resized image" in effective
    assert "896 x 1288" in effective


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
    Image.new("RGB", (32, 48), color=(255, 255, 255)).save(target_image)
    Image.new("RGB", (20, 12), color=(240, 240, 240)).save(example_image)

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

    class _FakeMediaResolution:
        MEDIA_RESOLUTION_HIGH = "HIGH"

    class _FakeTypes:
        Part = _FakePart
        ThinkingConfig = _FakeThinkingConfig
        GenerateContentConfig = _FakeGenerateContentConfig
        ThinkingLevel = _FakeThinkingLevel
        MediaResolution = _FakeMediaResolution

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
    example_payload = _prompt_only_payload(
        image="example.png",
        facts=[
            {
                "bbox": [1, 2, 3, 4],
                "fact_num": 1,
                "value": "10",
                "equations": None,
                "value_type": None,
                "value_context": None,
                "natural_sign": "positive",
                "row_role": "detail",
                "currency": None,
                "scale": None,
                "period_type": None,
                "period_start": None,
                "period_end": None,
                "duration_type": None,
                "recurring_period": None,
                "note_flag": False,
                "note_num": None,
                "note_name": None,
                "path": [],
                "path_source": None,
                "note_ref": None,
                "comment_ref": None,
            }
        ],
    )

    text = gemini_vlm.generate_content_from_image(
        image_path=target_image,
        prompt="Current image size: 32 x 48 pixels.\nextract now",
        model="Gemini Pro",
        api_key="k",
        system_prompt="system exact",
        few_shot_examples=[
            {
                "image_path": example_image,
                "expected_json": json.dumps(example_payload, ensure_ascii=False, separators=(",", ":")),
            }
        ],
        temperature=0.35,
        enable_thinking=True,
        thinking_level="high",
        response_mime_type="application/json",
        media_resolution="high",
    )

    assert text == '{"ok":true}'
    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    request_payload = json.loads((log_dir / "request.json").read_text(encoding="utf-8"))
    response_payload = json.loads((log_dir / "response.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert request_payload["model"] == "gemini-3.1-pro-preview"
    assert request_payload["prompt"] == "Current image size: 32 x 48 pixels.\nextract now"
    assert request_payload["system_prompt"] == "system exact"
    assert request_payload["image_summary"]["image_width"] == 32
    assert request_payload["image_summary"]["image_height"] == 48
    assert request_payload["image_summary"]["mime_type"] == "image/png"
    assert request_payload["prompt_image_size"] == {
        "mentions_current_image_size": True,
        "prompt_image_size_text": "32 x 48 pixels",
        "prompt_image_width": 32,
        "prompt_image_height": 48,
        "matches_source_image": True,
    }
    assert request_payload["temperature"] == 0.35
    assert request_payload["request_summary"]["model_family"] == "gemini_3"
    assert request_payload["request_summary"]["thinking_semantics"] == "thinking_level"
    assert request_payload["request_summary"]["effective_thinking_value"] == "HIGH"
    assert request_payload["request_summary"]["requested_temperature"] == 0.35
    assert request_payload["request_summary"]["effective_temperature"] == 0.35
    assert request_payload["exact_request"]["config"]["kwargs"]["temperature"] == 0.35
    assert request_payload["exact_request"]["config"]["kwargs"]["system_instruction"] == "system exact"
    assert request_payload["exact_request"]["config"]["kwargs"]["response_mime_type"] == "application/json"
    assert request_payload["exact_request"]["config"]["kwargs"]["media_resolution"] == "HIGH"
    assert request_payload["exact_request"]["contents"][0]["role"] == "user"
    assert request_payload["exact_request"]["contents"][0]["parts"][0] == {"type": "image_file", "file": "few_shot_01_example.png"}
    assert request_payload["exact_request"]["contents"][-1]["parts"][0] == {"type": "image_file", "file": "input_target.png"}
    assert request_payload["few_shot_examples"][0]["expected_json"] == json.dumps(
        example_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert request_payload["few_shot_examples"][0]["image_summary"]["image_width"] == 20
    assert request_payload["few_shot_examples"][0]["image_summary"]["image_height"] == 12
    assert request_payload["few_shot_examples"][0]["schema_signature"]["top_level_keys"] == ["pages"]
    assert request_payload["few_shot_examples"][0]["schema_signature"]["page_meta_keys"] == [
        "entity_name",
        "page_num",
        "page_type",
        "statement_type",
        "title",
    ]
    assert request_payload["few_shot_examples"][0]["schema_signature"]["fact_keys"] == [
        "bbox",
        "fact_num",
        "value",
        "equations",
        "value_type",
        "value_context",
        "natural_sign",
        "row_role",
        "currency",
        "scale",
        "period_type",
        "period_start",
        "period_end",
        "duration_type",
        "recurring_period",
        "note_flag",
        "note_num",
        "note_name",
        "path",
        "path_source",
        "note_ref",
        "comment_ref",
    ]
    assert (log_dir / request_payload["logged_image_path"]).is_file()
    assert (log_dir / request_payload["few_shot_examples"][0]["logged_image_path"]).is_file()
    assert (log_dir / "output.txt").read_text(encoding="utf-8") == '{"ok":true}'
    assert response_payload["thinking"] == "hidden-thought"
    assert any(event["event"] == "thinking_request_summary" for event in events)
    thinking_response = next(event for event in events if event["event"] == "thinking_response_summary")
    assert thinking_response["observed_thinking_signal"] is True


def test_generate_content_from_image_uses_vertex_endpoint_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    Image.new("RGB", (2000, 1000), color=(255, 255, 255)).save(target_image)
    _install_fake_qwen_vl_utils(monkeypatch, height=756, width=1512)

    seen: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        text = '{"candidates":[{"content":{"parts":[{"text":"{\\"ok\\":true}"}]}}]}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        _wrapper_payload(
                                            facts=[
                                                {"bbox": [100, 50, 20, 10], "value": "10", "path": []},
                                            ]
                                        ),
                                        ensure_ascii=False,
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

    class _FakeClient:
        def __init__(self, timeout: float):
            seen["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def post(self, endpoint: str, *, headers: dict[str, str], json: dict[str, object]):
            seen["endpoint"] = endpoint
            seen["headers"] = headers
            seen["json"] = json
            return _FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(gemini_vlm, "_resolve_vertex_generate_content_url", lambda **_kwargs: "https://vertex.example/generateContent")
    monkeypatch.setattr(gemini_vlm, "_resolve_vertex_access_token", lambda _token=None: "vertex-token")

    text = gemini_vlm.generate_content_from_image(
        image_path=target_image,
        prompt="extract now",
        model="gemini-flash-hf-tuned",
        system_prompt="system exact",
        temperature=0.2,
        enable_thinking=True,
        thinking_level="high",
    )

    payload = json.loads(text)
    assert payload["pages"][0]["facts"][0]["bbox"] == [132.28, 66.14, 26.46, 13.23]
    assert seen["endpoint"] == "https://vertex.example/generateContent"
    assert seen["headers"] == {
        "Authorization": "Bearer vertex-token",
        "Content-Type": "application/json",
    }
    request_payload = seen["json"]
    assert request_payload["contents"][0]["role"] == "user"
    assert request_payload["systemInstruction"] == {"parts": [{"text": "system exact"}]}
    assert request_payload["generationConfig"] == {"temperature": 0.2}
    assert request_payload["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "image/png"
    assert isinstance(request_payload["contents"][0]["parts"][0]["inlineData"]["data"], str)
    assert "resized image" in request_payload["contents"][0]["parts"][1]["text"]
    assert "extract now" in request_payload["contents"][0]["parts"][1]["text"]

    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    request_log = json.loads((log_dir / "request.json").read_text(encoding="utf-8"))
    assert request_log["backend"] == "vertex_generate_content"
    assert request_log["system_prompt"] == "system exact"
    assert request_log["temperature"] == 0.2
    assert request_log["request_summary"]["requested_temperature"] == 0.2
    assert request_log["image_resize"] == {
        "max_pixels": gemini_vlm.DEFAULT_VERTEX_TUNED_MAX_PIXELS,
        "original_width": 2000,
        "original_height": 1000,
        "prepared_width": 1512,
        "prepared_height": 756,
        "resized": True,
    }
    assert request_log["exact_request"]["endpoint"] == "https://vertex.example/generateContent"
    assert request_log["exact_request"]["json"]["contents"][0]["parts"][0]["inlineData"] == {"file": "input_target.png"}


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


def test_stream_content_from_image_uses_vertex_stream_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    Image.new("RGB", (2000, 1000), color=(255, 255, 255)).save(target_image)
    _install_fake_qwen_vl_utils(monkeypatch, height=756, width=1512)

    seen: dict[str, object] = {}

    class _FakeStreamResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_text(self):
            yield json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": '{"meta":{"entity_name":"A"},"facts":['},
                                ]
                            }
                        }
                    ]
                }
            )
            yield json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": '{"bbox":[100,50,20,10],"value":"10","path":[]}]}'},
                                ]
                            }
                        }
                    ]
                }
            )

    class _FakeClient:
        def __init__(self, timeout: float):
            seen["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def stream(self, method: str, endpoint: str, *, headers: dict[str, str], json: dict[str, object]):
            seen["method"] = method
            seen["endpoint"] = endpoint
            seen["headers"] = headers
            seen["json"] = json
            return _FakeStreamResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(gemini_vlm, "_resolve_vertex_stream_generate_content_url", lambda **_kwargs: "https://vertex.example/streamGenerateContent")
    monkeypatch.setattr(gemini_vlm, "_resolve_vertex_access_token", lambda _token=None: "vertex-token")

    chunks = list(
        gemini_vlm.stream_content_from_image(
            image_path=target_image,
            prompt="extract stream",
            model="gemini-flash-hf-tuned",
            system_prompt="system exact",
            temperature=0.15,
            enable_thinking=True,
            thinking_level="high",
        )
    )

    assert chunks == ['{"meta":{"entity_name":"A"},"facts":[', '{"bbox":[132.28, 66.14, 26.46, 13.23],"value":"10","path":[]}]}']
    assert seen["method"] == "POST"
    assert seen["endpoint"] == "https://vertex.example/streamGenerateContent"
    assert seen["headers"] == {
        "Authorization": "Bearer vertex-token",
        "Content-Type": "application/json",
    }
    request_payload = seen["json"]
    assert request_payload["systemInstruction"] == {"parts": [{"text": "system exact"}]}
    assert request_payload["generationConfig"] == {"temperature": 0.15}
    assert "resized image" in request_payload["contents"][0]["parts"][1]["text"]
    assert "extract stream" in request_payload["contents"][0]["parts"][1]["text"]

    log_dirs = list((tmp_path / "gemini_logs").iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    event_lines = (log_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    event_payloads = [json.loads(line) for line in event_lines]
    chunk_events = [payload for payload in event_payloads if payload["event"] == "stream_chunk"]
    request_log = json.loads((log_dir / "request.json").read_text(encoding="utf-8"))

    assert len(chunk_events) == 2
    assert request_log["backend"] == "vertex_stream_generate_content"
    assert request_log["system_prompt"] == "system exact"
    assert request_log["temperature"] == 0.15
    assert request_log["request_summary"]["requested_temperature"] == 0.15
    assert request_log["exact_request"]["endpoint"] == "https://vertex.example/streamGenerateContent"
    assert (log_dir / "output.txt").read_text(encoding="utf-8") == "".join(chunks)


def test_stream_content_from_image_vertex_rescales_bbox_split_across_chunks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target_image = tmp_path / "target.png"
    Image.new("RGB", (2000, 1000), color=(255, 255, 255)).save(target_image)
    _install_fake_qwen_vl_utils(monkeypatch, height=756, width=1512)

    class _FakeStreamResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_text(self):
            yield json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": '{"meta":{"entity_name":"A"},"facts":[{"bbox":[100,'},
                                ]
                            }
                        }
                    ]
                }
            )
            yield json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": '50,20,10],"value":"10","path":[]}]}'},
                                ]
                            }
                        }
                    ]
                }
            )

    class _FakeClient:
        def __init__(self, timeout: float):
            _ = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def stream(self, method: str, endpoint: str, *, headers: dict[str, str], json: dict[str, object]):
            _ = method, endpoint, headers, json
            return _FakeStreamResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(gemini_vlm, "_resolve_vertex_stream_generate_content_url", lambda **_kwargs: "https://vertex.example/streamGenerateContent")
    monkeypatch.setattr(gemini_vlm, "_resolve_vertex_access_token", lambda _token=None: "vertex-token")

    chunks = list(
        gemini_vlm.stream_content_from_image(
            image_path=target_image,
            prompt="extract stream",
            model="gemini-flash-hf-tuned",
            enable_thinking=False,
        )
    )

    combined = "".join(chunks)

    assert '{"bbox":[100,50,20,10]' not in combined
    assert '{"bbox":[132.28, 66.14, 26.46, 13.23],"value":"10","path":[]}]}' in combined


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


def test_analyze_logged_bbox_session_flags_tracked_bad_log_and_writes_overlay(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "gemini_logs/20260317T205644.557610Z_stream_content_gemini-3-flash-preview"
    session_dir = tmp_path / "bad_session"
    shutil.copytree(source_dir, session_dir)

    diagnostics = gemini_vlm.analyze_logged_bbox_session(session_dir)
    request_payload = json.loads((session_dir / "request.json").read_text(encoding="utf-8"))

    assert diagnostics["suspicious"] is True
    assert "wide_low_ink_strips" in diagnostics["suspicion_reasons"]
    assert diagnostics["preferred_bbox_mode"] == "normalized_1000_to_pixel"
    assert diagnostics["bbox_mode_scores"]["normalized_1000_to_pixel"] > diagnostics["bbox_mode_scores"]["pixel_as_is"]
    assert diagnostics["overlay_path"] == "bbox_overlay.png"
    assert (session_dir / "bbox_overlay.png").is_file()
    assert request_payload["bbox_diagnostics"]["suspicious"] is True
    assert request_payload["bbox_diagnostics"]["preferred_bbox_mode"] == "normalized_1000_to_pixel"
    assert "wide_low_ink_strips" in request_payload["bbox_diagnostics"]["suspicion_reasons"]


def test_analyze_bbox_response_accepts_known_good_2015_page(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "data/pdf_images/2015/page_0011.png"
    annotations = json.loads((repo_root / "data/annotations/2015.json").read_text(encoding="utf-8"))
    page_payload = next(page for page in annotations["pages"] if page.get("image") == "page_0011.png")
    overlay_path = tmp_path / "bbox_overlay.png"

    diagnostics = gemini_vlm.analyze_bbox_response(
        image_path,
        json.dumps({"pages": [page_payload]}, ensure_ascii=False),
        overlay_path=overlay_path,
    )

    assert diagnostics["bbox_count"] > 0
    assert diagnostics["suspicious"] is False
    assert diagnostics["preferred_bbox_mode"] == "pixel_as_is"
    assert diagnostics["bbox_mode_scores"]["pixel_as_is"] > diagnostics["bbox_mode_scores"]["normalized_1000_to_pixel"]
    assert diagnostics["median_aspect_ratio"] < 10.0
    assert diagnostics["overlay_path"] is None
    assert overlay_path.exists() is False


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


def test_parse_page_extraction_text_clears_note_flags_on_non_notes_pages() -> None:
    payload = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            },
            facts=[
                {
                    "bbox": [1, 2, 3, 4],
                    "value": "10",
                    "note_flag": True,
                    "note_num": 7,
                    "note_ref": None,
                    "path": [],
                }
            ],
        )
    )

    extraction = gemini_vlm.parse_page_extraction_text(payload)

    assert extraction.meta.statement_type.value == "income_statement"
    assert extraction.facts[0].note_flag is False
    assert extraction.facts[0].note_num is None


def test_parse_page_extraction_text_auto_resolves_normalized_bboxes_when_image_path_is_provided(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (800, 800), color=(255, 255, 255)).save(image_path)
    image = Image.open(image_path)
    for x in range(400, 560):
        for y in range(400, 480):
            image.putpixel((x, y), (16, 16, 16))
    image.save(image_path)

    payload = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            },
            facts=[
                {"bbox": [500, 500, 200, 100], "value": "10", "note_ref": None, "path": []},
            ],
        )
    )

    extraction = gemini_vlm.parse_page_extraction_text(payload, image_path=image_path)

    assert extraction.facts[0].bbox.x == pytest.approx(400.0)
    assert extraction.facts[0].bbox.y == pytest.approx(400.0)
    assert extraction.facts[0].bbox.w == pytest.approx(160.0)
    assert extraction.facts[0].bbox.h == pytest.approx(80.0)


def test_parse_page_extraction_text_uses_page_wide_normalized_mode_when_global_score_dominates(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (800, 800), color=(255, 255, 255)).save(image_path)
    image = Image.open(image_path)
    normalized_rects = [
        (400, 400, 80, 40),
        (520, 400, 80, 40),
        (400, 520, 80, 40),
    ]
    for rect in normalized_rects:
        x0, y0, w, h = rect
        for x in range(x0, x0 + w):
            for y in range(y0, y0 + h):
                image.putpixel((x, y), (16, 16, 16))
    for x in range(500, 600):
        for y in range(500, 550):
            image.putpixel((x, y), (16, 16, 16))
    image.save(image_path)

    payload = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            },
            facts=[
                {"bbox": [500, 500, 100, 50], "value": "10", "note_ref": None, "path": []},
                {"bbox": [650, 500, 100, 50], "value": "11", "note_ref": None, "path": []},
                {"bbox": [500, 650, 100, 50], "value": "12", "note_ref": None, "path": []},
            ],
        )
    )

    normalized = gemini_vlm._normalize_page_extraction_payload(json.loads(payload))
    resolved, mode, scores, _fact_modes, _policy = gemini_vlm._resolve_page_extraction_bbox_mode(
        normalized,
        image_path=image_path,
    )

    assert mode == "normalized_1000_to_pixel"
    assert scores["normalized_1000_to_pixel"] > scores["pixel_as_is"]
    for fact, (expected_x, expected_y, expected_w, expected_h) in zip(resolved["pages"][0]["facts"], normalized_rects):
        assert fact["bbox"][0] == pytest.approx(expected_x)
        assert fact["bbox"][1] == pytest.approx(expected_y)
        assert fact["bbox"][2] == pytest.approx(expected_w)
        assert fact["bbox"][3] == pytest.approx(expected_h)


def test_streaming_parser_uses_meta_to_clear_note_flags_on_non_notes_pages() -> None:
    parser = gemini_vlm.StreamingPageExtractionParser()
    meta_chunk = '{"meta":{"entity_name":null,"page_num":"3","page_type":"statements","statement_type":"income_statement","title":null},'
    facts_chunk = '"facts":[{"bbox":[1,2,3,4],"value":"10","note_flag":true,"note_num":7,"note_ref":null,"path":[]}]}'

    meta, facts = parser.feed(meta_chunk)
    assert meta is not None
    assert facts == []

    meta2, facts2 = parser.feed(facts_chunk)
    assert meta2 is None
    assert len(facts2) == 1
    assert facts2[0]["note_flag"] is False
    assert facts2[0]["note_num"] is None

    extraction = parser.finalize()
    assert extraction.facts[0].note_flag is False
    assert extraction.facts[0].note_num is None


def test_streaming_parser_finalize_auto_resolves_normalized_bboxes_when_image_path_is_provided(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (800, 800), color=(255, 255, 255)).save(image_path)
    image = Image.open(image_path)
    for x in range(400, 560):
        for y in range(400, 480):
            image.putpixel((x, y), (16, 16, 16))
    image.save(image_path)

    parser = gemini_vlm.StreamingPageExtractionParser(image_path=image_path)
    chunk = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            },
            facts=[{"bbox": [500, 500, 200, 100], "value": "10", "note_ref": None, "path": []}],
        )
    )
    parser.feed(chunk)

    extraction = parser.finalize()

    assert extraction.facts[0].bbox.x == pytest.approx(400.0)
    assert extraction.facts[0].bbox.y == pytest.approx(400.0)
    assert extraction.facts[0].bbox.w == pytest.approx(160.0)
    assert extraction.facts[0].bbox.h == pytest.approx(80.0)


def test_streaming_parser_finalize_clears_partial_duration_period_range() -> None:
    parser = gemini_vlm.StreamingPageExtractionParser()
    chunk = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "3",
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            },
            facts=[
                {
                    "bbox": [10, 20, 30, 40],
                    "value": "10",
                    "period_type": "duration",
                    "period_start": "2024-01-01",
                    "period_end": None,
                    "path": [],
                }
            ],
        )
    )
    parser.feed(chunk)

    extraction = parser.finalize()

    assert extraction.facts[0].period_type is not None
    assert extraction.facts[0].period_type.value == "duration"
    assert extraction.facts[0].period_start is None
    assert extraction.facts[0].period_end is None


def test_parse_page_extraction_text_drops_invalid_facts_and_writes_issue_summary(tmp_path: Path) -> None:
    session_dir, image_path = _create_logged_session(tmp_path)
    raw_text = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "11",
                "page_type": "statements",
                "statement_type": "notes_to_financial_statements",
                "title": "Notes",
            },
            facts=[
                {
                    "bbox": [100, 120, 40, 20],
                    "value": "10",
                    "note_flag": True,
                    "note_num": 5,
                    "path": ["A"],
                },
                {
                    "bbox": [180, 120, 40, 20],
                    "value": "20",
                    "note_flag": False,
                    "note_num": 5,
                    "path": ["B"],
                },
            ],
        ),
        ensure_ascii=False,
    )

    extraction = gemini_vlm.parse_page_extraction_text(
        raw_text,
        image_path=image_path,
        session_dir=session_dir,
    )
    summary = gemini_vlm.load_issue_summary(session_dir)
    trace_rows = [
        json.loads(line)
        for line in (session_dir / "fact_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(extraction.facts) == 1
    assert extraction.facts[0].value == "10"
    assert summary is not None
    assert summary["status"] == "warning"
    assert summary["kept_fact_count"] == 1
    assert summary["dropped_fact_count"] == 1
    assert summary["validation_failure_groups"][0]["code"] == "note_num_without_note_flag"
    assert summary["validation_failure_groups"][0]["count"] == 1
    assert any(row["source_stage"] == "final_validated" for row in trace_rows)
    assert any(row["source_stage"] == "final_dropped" for row in trace_rows)


def test_streaming_parser_records_fact_lineage_across_stream_and_finalize(tmp_path: Path) -> None:
    session_dir, image_path = _create_logged_session(tmp_path)
    parser = gemini_vlm.StreamingPageExtractionParser(image_path=image_path, session_dir=session_dir)
    payload = json.dumps(
        _wrapper_payload(
            meta={
                "entity_name": None,
                "page_num": "11",
                "page_type": "statements",
                "statement_type": "notes_to_financial_statements",
                "title": "Notes",
            },
            facts=[
                {
                    "bbox": [100, 120, 40, 20],
                    "value": "10",
                    "note_flag": True,
                    "note_num": 5,
                    "path": ["A"],
                },
                {
                    "bbox": [180, 120, 40, 20],
                    "value": "20",
                    "note_flag": False,
                    "note_num": 5,
                    "path": ["B"],
                },
            ],
        ),
        ensure_ascii=False,
    )

    parser.feed(payload)
    extraction = parser.finalize()
    trace_rows = [
        json.loads(line)
        for line in (session_dir / "fact_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = {row["source_stage"] for row in trace_rows}

    assert len(extraction.facts) == 1
    assert stages >= {
        "stream_partial",
        "stream_normalized",
        "final_parse",
        "final_resolved",
        "final_validated",
        "final_dropped",
    }


def test_parse_page_extraction_text_latest_bad_log_writes_grouped_issue_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "gemini_logs/20260319T115800.605745Z_stream_content_gemini-3-flash-preview"
    session_dir = tmp_path / "bad_note_session"
    shutil.copytree(source_dir, session_dir)
    raw_text = (session_dir / "output.txt").read_text(encoding="utf-8")
    image_path = session_dir / "input_target.png"

    with pytest.raises(ValueError, match=r"Kept 0 valid fact\(s\), dropped 24 invalid fact\(s\)\."):
        gemini_vlm.parse_page_extraction_text(
            raw_text,
            image_path=image_path,
            session_dir=session_dir,
        )

    summary = gemini_vlm.load_issue_summary(session_dir)

    assert summary is not None
    assert summary["status"] == "error"
    assert summary["kept_fact_count"] == 0
    assert summary["dropped_fact_count"] == 24
    assert summary["validation_failure_groups"][0]["code"] == "note_num_without_note_flag"
    assert summary["validation_failure_groups"][0]["count"] == 24
    assert (session_dir / "issue_summary.json").is_file()
    assert (session_dir / "fact_trace.jsonl").is_file()


def test_parse_page_extraction_text_records_raw_and_resolved_bbox_diagnostics_for_suspicious_log(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "gemini_logs/20260317T205644.557610Z_stream_content_gemini-3-flash-preview"
    session_dir = tmp_path / "bbox_session"
    shutil.copytree(source_dir, session_dir)
    gemini_vlm.analyze_logged_bbox_session(session_dir)
    raw_text = (session_dir / "output.txt").read_text(encoding="utf-8")
    image_path = session_dir / "input_target.png"

    extraction = gemini_vlm.parse_page_extraction_text(
        raw_text,
        image_path=image_path,
        session_dir=session_dir,
    )
    summary = gemini_vlm.load_issue_summary(session_dir)

    assert len(extraction.facts) > 0
    assert summary is not None
    assert summary["bbox_diagnostics_raw"]["suspicious"] is True
    assert summary["bbox_diagnostics_resolved"]["bbox_count"] == len(extraction.facts)
    assert (session_dir / "bbox_overlay.png").is_file()
    assert (session_dir / "bbox_overlay_resolved.png").is_file()


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


def test_parse_selected_field_patch_text_accepts_value_and_structural_updates() -> None:
    payload = {
        "meta_updates": {},
        "fact_updates": [
            {
                "fact_num": 1,
                "updates": {
                    "value": "(123)",
                    "note_flag": True,
                    "note_num": 7,
                    "path": ["assets", "current"],
                },
            }
        ],
    }
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(payload),
        allowed_fact_fields={"value", "note_flag", "note_num", "path"},
        allow_statement_type=False,
    )

    assert parsed["fact_updates"] == [
        {
            "fact_num": 1,
            "updates": {
                "value": "(123)",
                "note_flag": True,
                "note_num": 7,
                "path": ["assets", "current"],
            },
        }
    ]


def test_parse_selected_field_patch_text_drops_blank_string_updates() -> None:
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(
            {
                "meta_updates": {},
                "fact_updates": [
                    {
                        "fact_num": 1,
                        "updates": {
                            "value": "",
                            "path": "   ",
                            "period_type": "instant",
                        },
                    }
                ],
            }
        ),
        allowed_fact_fields={"value", "path", "period_type"},
        allow_statement_type=False,
    )

    assert parsed["fact_updates"] == [
        {
            "fact_num": 1,
            "updates": {
                "period_type": "instant",
            },
        }
    ]


def test_parse_selected_field_patch_text_ignores_blank_unrequested_fields() -> None:
    parsed = gemini_vlm.parse_selected_field_patch_text(
        json.dumps(
            {
                "meta_updates": {},
                "fact_updates": [
                    {
                        "fact_num": 1,
                        "updates": {
                            "period_type": "duration",
                            "natural_sign": "",
                        },
                    }
                ],
            }
        ),
        allowed_fact_fields={"period_type"},
        allow_statement_type=False,
    )

    assert parsed["fact_updates"] == [
        {
            "fact_num": 1,
            "updates": {
                "period_type": "duration",
            },
        }
    ]
