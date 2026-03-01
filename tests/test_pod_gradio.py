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
