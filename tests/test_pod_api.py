from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from fastapi import HTTPException

from finetree_annotator.deploy import pod_api


def test_extract_prompt_and_image_url_supports_openai_multimodal_shape() -> None:
    prompt, image_url = pod_api._extract_prompt_and_image_url(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        }
    )
    assert prompt == "Describe this"
    assert image_url == "data:image/png;base64,AAAA"


def test_extract_prompt_and_image_url_rejects_missing_image() -> None:
    with pytest.raises(ValueError):
        pod_api._extract_prompt_and_image_url(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            }
        )


def test_image_path_from_data_uri_writes_file(tmp_path: Path) -> None:
    path = pod_api._image_path_from_data_uri("data:image/png;base64,AAAA", temp_dir=tmp_path)
    assert path.is_file()
    assert path.read_bytes() == b"\x00\x00\x00"


def test_bearer_token_parsing() -> None:
    assert pod_api._bearer_token("Bearer abc") == "abc"
    assert pod_api._bearer_token("bearer xyz") == "xyz"
    assert pod_api._bearer_token("Token xyz") == ""


def test_resolve_model_selection_uses_served_alias_by_default() -> None:
    response_model, inference_model = pod_api._resolve_model_selection(None, "qwen-gt")
    assert response_model == "qwen-gt"
    assert inference_model is None


def test_resolve_model_selection_ignores_served_alias_override() -> None:
    response_model, inference_model = pod_api._resolve_model_selection("qwen-gt", "qwen-gt")
    assert response_model == "qwen-gt"
    assert inference_model is None


def test_resolve_model_selection_accepts_explicit_underlying_model() -> None:
    response_model, inference_model = pod_api._resolve_model_selection("unsloth/Qwen3.5-35B-A3B", "qwen-gt")
    assert response_model == "unsloth/Qwen3.5-35B-A3B"
    assert inference_model == "unsloth/Qwen3.5-35B-A3B"


def test_chat_completions_route_uses_body_payload_not_query_request() -> None:
    app = pod_api.create_app(api_key="test-key")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    query_param_names = [p.name for p in route.dependant.query_params]
    body_param_names = [p.name for p in route.dependant.body_params]
    assert "request" not in query_param_names
    assert "payload" in body_param_names


def test_openapi_generation_includes_chat_completions() -> None:
    app = pod_api.create_app(api_key="test-key")
    spec = app.openapi()
    assert "/v1/chat/completions" in spec["paths"]


def test_chat_completions_passes_max_tokens_to_inference(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_generate_content_from_image(**kwargs) -> str:
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(pod_api, "generate_content_from_image", _fake_generate_content_from_image)
    app = pod_api.create_app(api_key="test-key")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    image_data_uri = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")

    response = asyncio.run(
        route.endpoint(
            payload={
                "model": "qwen-gt",
                "max_tokens": 13,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    }
                ],
            },
            authorization="Bearer test-key",
        )
    )

    assert response.status_code == 200
    assert seen["max_new_tokens"] == 13


def test_chat_completions_internal_error_returns_error_id(monkeypatch) -> None:
    def _fake_generate_content_from_image(**kwargs) -> str:
        raise RuntimeError("boom")

    monkeypatch.delenv("FINETREE_POD_DEBUG_ERRORS", raising=False)
    monkeypatch.setattr(pod_api, "generate_content_from_image", _fake_generate_content_from_image)
    app = pod_api.create_app(api_key="test-key")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    image_data_uri = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            route.endpoint(
                payload={
                    "model": "qwen-gt",
                    "max_tokens": 13,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello"},
                                {"type": "image_url", "image_url": {"url": image_data_uri}},
                            ],
                        }
                    ],
                },
                authorization="Bearer test-key",
            )
        )

    assert exc_info.value.status_code == 500
    assert str(exc_info.value.detail).startswith("Internal server error. error_id=poderr-")
    assert "detail=" not in str(exc_info.value.detail)


def test_chat_completions_debug_error_includes_exception_detail(monkeypatch) -> None:
    def _fake_generate_content_from_image(**kwargs) -> str:
        raise RuntimeError("boom")

    monkeypatch.setenv("FINETREE_POD_DEBUG_ERRORS", "1")
    monkeypatch.setattr(pod_api, "generate_content_from_image", _fake_generate_content_from_image)
    app = pod_api.create_app(api_key="test-key")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    image_data_uri = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            route.endpoint(
                payload={
                    "model": "qwen-gt",
                    "max_tokens": 13,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello"},
                                {"type": "image_url", "image_url": {"url": image_data_uri}},
                            ],
                        }
                    ],
                },
                authorization="Bearer test-key",
            )
        )

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 500
    assert detail.startswith("Internal server error. error_id=poderr-")
    assert "detail=RuntimeError" in detail
    assert "boom" in detail
