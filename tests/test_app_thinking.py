from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from finetree_annotator import app as app_mod
from finetree_annotator import gemini_vlm, qwen_vlm


class _FakeParser:
    def feed(self, _chunk: str):
        return None, []

    def finalize(self):
        return SimpleNamespace(
            meta=SimpleNamespace(
                model_dump=lambda mode="json": {
                    "entity_name": None,
                    "page_num": None,
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                }
            ),
            facts=[],
        )


def test_gemini_stream_worker_forwards_enable_thinking(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_stream_content_from_image(**kwargs):
        seen.update(kwargs)
        yield (
            '{"images_dir":"data/pdf_images/doc1","metadata":{},'
            '"pages":[{"image":"page_0001.png","meta":{"entity_name":null,"page_num":null,'
            '"page_type":"other","statement_type":null,"title":null},"facts":[]}]}'
        )

    monkeypatch.setattr(gemini_vlm, "StreamingPageExtractionParser", _FakeParser)
    monkeypatch.setattr(gemini_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.GeminiStreamWorker(
        image_path=image_path,
        prompt="extract",
        model="gemini-3-flash-preview",
        api_key="k",
        enable_thinking=False,
    )

    worker.run()

    assert seen["enable_thinking"] is False


def test_gemini_stream_worker_forwards_model_and_thinking_level(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_stream_content_from_image(**kwargs):
        seen.update(kwargs)
        yield (
            '{"images_dir":"data/pdf_images/doc1","metadata":{},'
            '"pages":[{"image":"page_0001.png","meta":{"entity_name":null,"page_num":null,'
            '"page_type":"other","statement_type":null,"title":null},"facts":[]}]}'
        )

    monkeypatch.setattr(gemini_vlm, "StreamingPageExtractionParser", _FakeParser)
    monkeypatch.setattr(gemini_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.GeminiStreamWorker(
        image_path=image_path,
        prompt="extract",
        model="gemini-3.1-pro-preview",
        api_key="k",
        enable_thinking=True,
        thinking_level="high",
    )

    worker.run()

    assert seen["model"] == "gemini-3.1-pro-preview"
    assert seen["enable_thinking"] is True
    assert seen["thinking_level"] == "high"


def test_qwen_stream_worker_forwards_enable_thinking(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_stream_content_from_image(**kwargs):
        seen.update(kwargs)
        yield (
            '{"images_dir":"data/pdf_images/doc1","metadata":{},'
            '"pages":[{"image":"page_0001.png","meta":{"entity_name":null,"page_num":null,'
            '"page_type":"other","statement_type":null,"title":null},"facts":[]}]}'
        )

    monkeypatch.setattr(gemini_vlm, "StreamingPageExtractionParser", _FakeParser)
    monkeypatch.setattr(qwen_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.QwenStreamWorker(
        image_path=image_path,
        prompt="extract",
        model="local-qwen",
        config_path="cfg.yaml",
        enable_thinking=True,
    )

    worker.run()

    assert seen["enable_thinking"] is True


def test_gemini_fill_worker_forwards_enable_thinking(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_generate_content_from_image(**kwargs):
        seen.update(kwargs)
        return '{"meta_updates":{},"fact_updates":[]}'

    monkeypatch.setattr(gemini_vlm, "generate_content_from_image", _fake_generate_content_from_image)

    worker = app_mod.GeminiFillWorker(
        image_path=image_path,
        prompt="fill",
        model="gemini-3-flash-preview",
        api_key="k",
        allowed_fact_fields={"period_type"},
        allow_statement_type=False,
        enable_thinking=False,
    )

    worker.run()

    assert seen["enable_thinking"] is False
