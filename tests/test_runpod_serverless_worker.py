from __future__ import annotations

import base64
from pathlib import Path

import pytest

from finetree_annotator.deploy import runpod_serverless_worker as worker


def test_handler_page_extraction_mode_returns_jsonable_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake")

    seen: dict[str, str | None] = {}

    class _FakeExtraction:
        def model_dump(self) -> dict[str, str]:
            return {"status": "ok"}

    def _fake_generate_page_extraction_from_image(
        image_path: Path,
        prompt: str,
        model: str | None = None,
        config_path: str | None = None,
    ) -> _FakeExtraction:
        seen["image_path"] = str(image_path)
        seen["prompt"] = prompt
        seen["model"] = model
        seen["config_path"] = config_path
        return _FakeExtraction()

    monkeypatch.setattr(worker, "generate_page_extraction_from_image", _fake_generate_page_extraction_from_image)

    result = worker.handler(
        {
            "input": {
                "image_path": str(image_path),
                "prompt": "extract",
                "model": "asafd60/qwen35-test",
                "config_path": "configs/finetune_qwen35a3_vl_100gb_safe.yaml",
            }
        }
    )

    assert result == {"ok": True, "mode": "page_extraction", "result": {"status": "ok"}}
    assert seen["image_path"] == str(image_path.resolve())
    assert seen["prompt"] == "extract"
    assert seen["model"] == "asafd60/qwen35-test"
    assert seen["config_path"] == "configs/finetune_qwen35a3_vl_100gb_safe.yaml"


def test_run_inference_text_mode_with_base64_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str | None] = {}

    def _fake_generate_content_from_image(
        image_path: Path,
        prompt: str,
        model: str | None = None,
        config_path: str | None = None,
    ) -> str:
        seen["suffix"] = image_path.suffix
        seen["prompt"] = prompt
        seen["model"] = model
        seen["config_path"] = config_path
        return "hello"

    monkeypatch.setattr(worker, "generate_content_from_image", _fake_generate_content_from_image)

    payload = {
        "image_base64": base64.b64encode(b"abc").decode("ascii"),
        "image_mime_type": "image/jpeg",
        "response_mode": "text",
        "prompt": "caption",
        "model": "qwen35-test",
    }
    result = worker.run_inference(payload)

    assert result == {"ok": True, "mode": "text", "text": "hello"}
    assert seen["suffix"] == ".jpg"
    assert seen["prompt"] == "caption"
    assert seen["model"] == "qwen35-test"
    assert seen["config_path"] is None


def test_handler_rejects_non_object_input() -> None:
    with pytest.raises(TypeError):
        worker.handler({"input": "not-an-object"})  # type: ignore[arg-type]


def test_run_inference_rejects_invalid_response_mode(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake")
    with pytest.raises(ValueError):
        worker.run_inference({"image_path": str(image_path), "response_mode": "invalid"})


def test_stream_inference_text_mode_yields_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake")

    def _fake_stream_content_from_image(
        image_path: Path,
        prompt: str,
        model: str | None = None,
        config_path: str | None = None,
    ):
        yield "hello "
        yield "world"

    monkeypatch.setattr(worker, "stream_content_from_image", _fake_stream_content_from_image)

    chunks = list(
        worker.stream_inference(
            {
                "image_path": str(image_path),
                "response_mode": "text",
                "prompt": "p",
            }
        )
    )

    assert chunks == [
        {"ok": True, "mode": "text", "chunk": "hello "},
        {"ok": True, "mode": "text", "chunk": "world"},
    ]
