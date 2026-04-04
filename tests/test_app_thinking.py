from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from finetree_annotator import app as app_mod
from finetree_annotator import gemini_vlm, qwen_vlm


class _FakeParser:
    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs

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


def test_gemini_stream_worker_forwards_temperature(monkeypatch, tmp_path: Path) -> None:
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
        temperature=0.4,
        enable_thinking=False,
    )

    worker.run()

    assert seen["temperature"] == 0.4


def test_gemini_stream_worker_forwards_system_prompt(monkeypatch, tmp_path: Path) -> None:
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
        system_prompt="system exact",
        enable_thinking=False,
    )

    worker.run()

    assert seen["system_prompt"] == "system exact"


def test_gemini_stream_worker_uses_json_mode_and_high_media_resolution_for_gt(monkeypatch, tmp_path: Path) -> None:
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
        mode="gt",
        api_key="k",
        enable_thinking=False,
    )

    worker.run()

    assert seen["response_mime_type"] == "application/json"
    assert seen["media_resolution"] == "high"


def test_gemini_stream_worker_uses_json_mode_and_high_media_resolution_for_bbox_only(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_stream_content_from_image(**kwargs):
        seen.update(kwargs)
        yield '{"facts":[{"bbox":[10,20,30,40]}]}'

    class _BBoxOnlyParser:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def feed(self, _chunk: str):
            return None, []

        def finalize(self):
            return SimpleNamespace(meta={}, facts=[{"bbox": [10, 20, 30, 40]}])

    monkeypatch.setattr(gemini_vlm, "StreamingBBoxOnlyParser", _BBoxOnlyParser)
    monkeypatch.setattr(gemini_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.GeminiStreamWorker(
        image_path=image_path,
        prompt="extract bboxes",
        model="gemini-3-flash-preview",
        mode="bbox_only",
        api_key="k",
        enable_thinking=False,
    )

    worker.run()

    assert seen["response_mime_type"] == "application/json"
    assert seen["media_resolution"] == "high"


def test_gemini_stream_worker_stops_after_max_facts(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    def _fake_stream_content_from_image(**_kwargs):
        yield "chunk"

    class _LimitParser:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def feed(self, _chunk: str):
            facts = []
            for idx in range(5):
                facts.append(
                    {
                        "bbox": [10 + idx, 20, 5, 5],
                        "value": str(idx + 1),
                        "equation": None,
                        "value_type": None,
                        "value_context": None,
                        "natural_sign": "positive",
                        "row_role": "detail",
                        "currency": None,
                        "scale": None,
                        "date": None,
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
                )
            return {
                "entity_name": None,
                "page_num": None,
                "page_type": "other",
                "statement_type": None,
                "title": None,
            }, facts

        def finalize(self):
            raise AssertionError("finalize should not be called when max_facts stops the worker early")

    monkeypatch.setattr(gemini_vlm, "StreamingPageExtractionParser", _LimitParser)
    monkeypatch.setattr(gemini_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.GeminiStreamWorker(
        image_path=image_path,
        prompt="extract",
        model="gemini-3-flash-preview",
        api_key="k",
        enable_thinking=False,
        max_facts=3,
    )

    emitted_values: list[str] = []
    limit_hits: list[bool] = []
    completed_payloads: list[object] = []
    worker.fact_received.connect(lambda fact: emitted_values.append(str(fact["value"])))
    worker.limit_reached.connect(lambda: limit_hits.append(True))
    worker.completed.connect(lambda extraction: completed_payloads.append(extraction))

    worker.run()

    assert emitted_values == ["1", "2", "3"]
    assert limit_hits == [True]
    assert len(completed_payloads) == 1
    assert [fact.value for fact in completed_payloads[0].facts] == ["1", "2", "3"]


def test_gemini_stream_worker_allows_partial_finalize_error(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    def _fake_stream_content_from_image(**_kwargs):
        yield "chunk"

    class _FinalizeFailParser:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def feed(self, _chunk: str):
            return (
                {
                    "entity_name": None,
                    "page_num": None,
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                },
                [
                    {
                        "bbox": [10, 20, 30, 40],
                        "value": "100",
                        "row_role": "total",
                    }
                ],
            )

        def finalize(self):
            raise ValueError("total rows must contain at least one fact_equation reference.")

    monkeypatch.setattr(gemini_vlm, "StreamingPageExtractionParser", _FinalizeFailParser)
    monkeypatch.setattr(gemini_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.GeminiStreamWorker(
        image_path=image_path,
        prompt="extract",
        model="gemini-3-flash-preview",
        api_key="k",
        enable_thinking=False,
        allow_partial_finalize_error=True,
    )

    completed_payloads: list[object] = []
    failures: list[str] = []
    worker.completed.connect(lambda extraction: completed_payloads.append(extraction))
    worker.failed.connect(lambda message: failures.append(message))

    worker.run()

    assert not failures
    assert len(completed_payloads) == 1
    extraction = completed_payloads[0]
    fact_payload = extraction.facts[0].model_dump(mode="json")
    assert fact_payload["value"] == "100"
    assert fact_payload["row_role"] == "total"
    assert "fact_equation" not in fact_payload
    assert fact_payload.get("equations") is None


def test_gemini_stream_worker_finalize_error_without_partial_still_fails(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    def _fake_stream_content_from_image(**_kwargs):
        yield "chunk"

    class _FinalizeFailParser:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def feed(self, _chunk: str):
            return None, [{"bbox": [10, 20, 30, 40], "value": "100"}]

        def finalize(self):
            raise ValueError("boom")

    monkeypatch.setattr(gemini_vlm, "StreamingPageExtractionParser", _FinalizeFailParser)
    monkeypatch.setattr(gemini_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.GeminiStreamWorker(
        image_path=image_path,
        prompt="extract",
        model="gemini-3-flash-preview",
        api_key="k",
        enable_thinking=False,
        allow_partial_finalize_error=False,
    )

    completed_payloads: list[object] = []
    failures: list[str] = []
    worker.completed.connect(lambda extraction: completed_payloads.append(extraction))
    worker.failed.connect(lambda message: failures.append(message))

    worker.run()

    assert not completed_payloads
    assert failures == ["boom"]


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


def test_qwen_stream_worker_forwards_temperature(monkeypatch, tmp_path: Path) -> None:
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
        temperature=0.0,
        enable_thinking=False,
    )

    worker.run()

    assert seen["temperature"] == 0.0


def test_qwen_stream_worker_uses_bbox_only_parser_and_prepared_resize(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_stream_content_from_image(**kwargs):
        seen.update(kwargs)
        yield '{"facts":[{"bbox":[10,20,30,40],"value":"12"}]}'

    class _BBoxOnlyParser:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def feed(self, _chunk: str):
            return None, []

        def finalize(self):
            return SimpleNamespace(meta={}, facts=[{"bbox": [10, 20, 30, 40], "value": "12"}])

    monkeypatch.setattr(gemini_vlm, "StreamingBBoxOnlyParser", _BBoxOnlyParser)
    monkeypatch.setattr(qwen_vlm, "stream_content_from_image", _fake_stream_content_from_image)

    worker = app_mod.QwenStreamWorker(
        image_path=image_path,
        prompt="extract bbox values",
        model="local-qwen",
        mode="bbox_only",
        config_path="cfg.yaml",
        enable_thinking=False,
        prepared_size=(140, 84),
        original_size=(200.0, 120.0),
        bbox_max_pixels=1_400_000,
    )

    worker.run()

    assert seen["require_prepared_resize"] is True
    assert seen["prepared_size"] == (140, 84)
    assert seen["bbox_max_pixels"] == 1_400_000


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


def test_gemini_fill_worker_forwards_temperature(monkeypatch, tmp_path: Path) -> None:
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
        temperature=0.25,
    )

    worker.run()

    assert seen["temperature"] == 0.25


def test_qwen_fill_worker_forwards_temperature(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"img")

    seen: dict[str, object] = {}

    def _fake_generate_content_from_image(**kwargs):
        seen.update(kwargs)
        return '{"meta_updates":{},"fact_updates":[]}'

    monkeypatch.setattr(qwen_vlm, "generate_content_from_image", _fake_generate_content_from_image)

    worker = app_mod.QwenFillWorker(
        image_path=image_path,
        prompt="fill",
        model="local-qwen",
        config_path="cfg.yaml",
        allowed_fact_fields={"period_type"},
        allow_statement_type=False,
        temperature=0.0,
        enable_thinking=False,
    )

    worker.run()

    assert seen["temperature"] == 0.0
