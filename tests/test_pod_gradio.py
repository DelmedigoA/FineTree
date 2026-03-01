from __future__ import annotations

from pathlib import Path

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
