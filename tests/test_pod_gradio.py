from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from finetree_annotator.deploy import pod_gradio


def test_default_example_image_path_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"img")
    monkeypatch.setenv("FINETREE_GRADIO_EXAMPLE_IMAGE", str(image_path))
    resolved = pod_gradio._default_example_image_path()
    assert resolved == image_path.resolve()


def test_default_example_image_path_uses_preferred_repo_location(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    preferred = repo_root / "data" / "pdf_images" / "test" / "page_0001.png"
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_bytes(b"img")
    monkeypatch.delenv("FINETREE_GRADIO_EXAMPLE_IMAGE", raising=False)
    monkeypatch.setattr(pod_gradio, "_project_root", lambda: repo_root)
    resolved = pod_gradio._default_example_image_path()
    assert resolved == preferred.resolve()


def test_resolve_basic_auth_credentials_prefers_playground_env(monkeypatch) -> None:
    monkeypatch.setenv("FINETREE_PLAYGROUND_USER", "u1")
    monkeypatch.setenv("FINETREE_PLAYGROUND_PASS", "p1")
    monkeypatch.setenv("FINETREE_GRADIO_USER", "u2")
    monkeypatch.setenv("FINETREE_GRADIO_PASS", "p2")
    assert pod_gradio._resolve_basic_auth_credentials() == ("u1", "p1")


def test_build_openai_chat_payload_includes_system_and_sampling() -> None:
    payload = pod_gradio._build_openai_chat_payload(
        {
            "system_prompt": "be strict",
            "user_prompt": "hello",
            "model": "qwen-gt",
            "temperature": 0.4,
            "top_p": 0.8,
            "max_tokens": 33,
            "image_data_url": "data:image/png;base64,AAAA",
        }
    )
    assert payload["model"] == "qwen-gt"
    assert payload["max_tokens"] == 33
    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.8
    assert payload["messages"][0] == {"role": "system", "content": "be strict"}
    assert payload["messages"][1]["role"] == "user"


def test_build_openai_chat_payload_includes_recent_history_context() -> None:
    payload = pod_gradio._build_openai_chat_payload(
        {
            "system_prompt": "be strict",
            "user_prompt": "next question",
            "model": "qwen-gt",
            "max_tokens": 40,
            "history": [
                {"user": "first", "assistant": "first answer"},
                {"user": "second", "assistant": "second answer"},
            ],
            "image_data_url": "data:image/png;base64,AAAA",
        }
    )
    message_text = payload["messages"][1]["content"][0]["text"]
    assert "Conversation context:" in message_text
    assert "User: first" in message_text
    assert "Assistant: first answer" in message_text
    assert "User: next question" in message_text
    assert message_text.rstrip().endswith("Assistant:")


def test_iter_sse_data_events_parses_event_payloads(monkeypatch) -> None:
    stream_lines = [
        b": keepalive\n",
        b"event: message\n",
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(stream_lines)

    def _fake_urlopen(_req, timeout=None):
        assert timeout == 5.0
        return _FakeResponse()

    monkeypatch.setattr(pod_gradio.urllib_request, "urlopen", _fake_urlopen)
    events = list(
        pod_gradio._iter_sse_data_events(
            url="http://localhost/v1/chat/completions",
            body={"stream": True},
            api_key="token",
            timeout_sec=5.0,
        )
    )
    assert events == ['{"choices":[{"delta":{"content":"hi"}}]}', "[DONE]"]


def test_chat_stream_route_proxies_sse_and_sets_stream_flag(monkeypatch) -> None:
    monkeypatch.setenv("FINETREE_PLAYGROUND_USER", "u1")
    monkeypatch.setenv("FINETREE_PLAYGROUND_PASS", "p1")
    monkeypatch.setenv("FINETREE_POD_API_KEY", "pod-key")

    seen: dict[str, object] = {}

    def _fake_iter_sse_data_events(url: str, body: dict[str, object], api_key: str, timeout_sec: float):
        seen["url"] = url
        seen["body"] = body
        seen["api_key"] = api_key
        seen["timeout_sec"] = timeout_sec
        yield json.dumps({"choices": [{"delta": {"content": "hel"}}]})
        yield json.dumps({"choices": [{"delta": {"content": "lo"}}]})
        yield "[DONE]"

    monkeypatch.setattr(pod_gradio, "_iter_sse_data_events", _fake_iter_sse_data_events)

    app = pod_gradio.create_app(api_base_url="http://127.0.0.1:6666")
    client = TestClient(app)
    basic_token = base64.b64encode(b"u1:p1").decode("ascii")
    response = client.post(
        "/api/chat/stream",
        headers={"Authorization": f"Basic {basic_token}"},
        json={
            "system_prompt": "sys",
            "user_prompt": "hello",
            "model": "qwen-gt",
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 16,
            "image_data_url": "data:image/png;base64,AAAA",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body_text = response.text

    assert "data: {\"choices\": [{\"delta\": {\"content\": \"hel\"}}]}" in body_text
    assert "data: {\"choices\": [{\"delta\": {\"content\": \"lo\"}}]}" in body_text
    assert "data: [DONE]" in body_text
    assert seen["url"] == "http://127.0.0.1:6666/v1/chat/completions"
    assert seen["api_key"] == "pod-key"
    assert seen["body"]["stream"] is True


def test_playground_chat_routes_bind_payload_from_body(monkeypatch) -> None:
    monkeypatch.setenv("FINETREE_PLAYGROUND_USER", "u1")
    monkeypatch.setenv("FINETREE_PLAYGROUND_PASS", "p1")
    monkeypatch.setenv("FINETREE_POD_API_KEY", "pod-key")
    app = pod_gradio.create_app(api_base_url="http://127.0.0.1:6666")

    chat_route = next(r for r in app.routes if getattr(r, "path", "") == "/api/chat")
    stream_route = next(r for r in app.routes if getattr(r, "path", "") == "/api/chat/stream")

    assert "payload" in [p.name for p in chat_route.dependant.body_params]
    assert "payload" not in [p.name for p in chat_route.dependant.query_params]
    assert "payload" in [p.name for p in stream_route.dependant.body_params]
    assert "payload" not in [p.name for p in stream_route.dependant.query_params]


def test_history_messages_to_turns_converts_messages_shape() -> None:
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": {"text": "second answer"}},
    ]

    turns = pod_gradio._history_messages_to_turns(history)
    assert turns == [
        {"user": "first question", "assistant": "first answer"},
        {"user": "second question", "assistant": "second answer"},
    ]


def test_resolve_default_model_input_prefers_explicit_value() -> None:
    value = pod_gradio._resolve_default_model_input(
        default_model="Qwen/Qwen3.5-27B",
        config_path=None,
    )
    assert value == "Qwen/Qwen3.5-27B"
