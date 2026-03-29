from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import finetree_annotator.local_doctr as local_doctr
from finetree_annotator.local_doctr import (
    DEFAULT_DETECTOR_MODEL_PATH,
    DEFAULT_DOCTR_CACHE_DIR,
    DETECTOR_MODEL_ENV_VAR,
    FINE_TUNED_BBOX_SCALE,
    LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED,
    LOCAL_DETECTOR_BACKEND_FINE_TUNED,
    LOCAL_DETECTOR_BACKEND_MERGED,
    LOCAL_DETECTOR_MODEL_NAME,
    MATCH_INTERSECTION_OVER_OLD_THRESHOLD,
    MATCH_IOU_THRESHOLD,
    MERGED_DETECTOR_MODEL_NAME,
    STOCK_DOCTR_DETECTOR_MODEL_NAME,
    detector_checkpoint_missing_message,
    detector_word_to_bbox,
    doctr_geometry_to_bbox,
    extract_numeric_bbox_candidates,
    extract_numeric_bbox_facts,
    is_excluded_numeric_token_text,
    is_numeric_token_text,
    local_detector_model_name,
    normalize_fine_tuned_bbox,
    normalize_local_detector_backend,
    pretrained_detector_unavailable_message,
    resolve_detector_model_path,
)


def test_is_numeric_token_text_accepts_supported_numeric_forms() -> None:
    assert is_numeric_token_text("-")
    assert is_numeric_token_text("1,234")
    assert is_numeric_token_text("(123)")
    assert is_numeric_token_text("₪42.5")
    assert is_numeric_token_text("12%")
    assert not is_numeric_token_text("31")
    assert not is_numeric_token_text("2007")
    assert not is_numeric_token_text("2026")
    assert not is_numeric_token_text("Assets")
    assert not is_numeric_token_text("12x")


def test_is_excluded_numeric_token_text_rejects_year_range_and_31() -> None:
    assert is_excluded_numeric_token_text("31")
    assert is_excluded_numeric_token_text("2000")
    assert is_excluded_numeric_token_text("2026")
    assert not is_excluded_numeric_token_text("1999")
    assert not is_excluded_numeric_token_text("2027")
    assert not is_excluded_numeric_token_text("32")


def test_doctr_geometry_to_bbox_converts_normalized_box_to_pixel_xywh() -> None:
    bbox = doctr_geometry_to_bbox(((0.10, 0.25), (0.35, 0.65)), image_width=200, image_height=100)
    assert bbox == [20, 25, 50, 40]


def test_doctr_geometry_to_bbox_rejects_zero_area() -> None:
    assert doctr_geometry_to_bbox(((0.25, 0.25), (0.25, 0.65)), image_width=200, image_height=100) is None


def test_detector_word_to_bbox_converts_normalized_xyxy_to_pixel_xywh() -> None:
    bbox = detector_word_to_bbox([0.10, 0.25, 0.35, 0.65, 0.9], image_width=200, image_height=100)
    assert bbox == [20, 25, 50, 40]


def test_normalize_fine_tuned_bbox_scales_box_to_33_percent_around_center() -> None:
    bbox = normalize_fine_tuned_bbox([20, 10, 40, 20], image_width=200, image_height=100)
    assert FINE_TUNED_BBOX_SCALE == 0.33
    assert bbox == [34, 16, 13, 7]


def test_resolve_detector_model_path_uses_default_and_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(DETECTOR_MODEL_ENV_VAR, raising=False)
    assert resolve_detector_model_path() == DEFAULT_DETECTOR_MODEL_PATH

    override = tmp_path / "custom.pt"
    monkeypatch.setenv(DETECTOR_MODEL_ENV_VAR, str(override))
    assert resolve_detector_model_path() == override.resolve()


def test_detector_checkpoint_missing_message_mentions_override_path() -> None:
    message = detector_checkpoint_missing_message(Path("/tmp/missing.pt"))
    assert "Resolved checkpoint path: /tmp/missing.pt" in message
    assert str(DEFAULT_DETECTOR_MODEL_PATH) in message
    assert DETECTOR_MODEL_ENV_VAR in message


def test_pretrained_detector_unavailable_message_mentions_cache_dir() -> None:
    message = pretrained_detector_unavailable_message(RuntimeError("offline"))
    assert str(DEFAULT_DOCTR_CACHE_DIR) in message
    assert "offline" in message


def test_local_detector_backend_helpers_map_to_expected_values() -> None:
    assert normalize_local_detector_backend("fine_tuned") == LOCAL_DETECTOR_BACKEND_FINE_TUNED
    assert normalize_local_detector_backend("doctr_pretrained") == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED
    assert normalize_local_detector_backend("merged") == LOCAL_DETECTOR_BACKEND_MERGED
    assert normalize_local_detector_backend("unknown") == LOCAL_DETECTOR_BACKEND_FINE_TUNED
    assert local_detector_model_name("fine_tuned") == LOCAL_DETECTOR_MODEL_NAME
    assert local_detector_model_name("doctr_pretrained") == STOCK_DOCTR_DETECTOR_MODEL_NAME
    assert local_detector_model_name("merged") == MERGED_DETECTOR_MODEL_NAME


def test_merge_threshold_constants_are_stricter_for_containment_than_iou() -> None:
    assert MATCH_INTERSECTION_OVER_OLD_THRESHOLD == 0.75
    assert MATCH_IOU_THRESHOLD == 0.25


def test_extract_numeric_bbox_candidates_uses_detector_then_crop_recognition(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def _fake_detector(_images):
        return [
            {
                "words": np.array(
                    [
                        [0.50, 0.20, 0.80, 0.40, 0.9],
                        [0.10, 0.10, 0.30, 0.30, 0.8],
                    ],
                    dtype=np.float32,
                )
            }
        ]

    monkeypatch.setattr(local_doctr, "_load_detector_for_backend", lambda _backend: _fake_detector)
    monkeypatch.setattr(
        local_doctr,
        "_recognize_crop_payloads",
        lambda crops, cancel_requested=None: [("55", 0.9), ("42", 0.8)],
    )

    facts = extract_numeric_bbox_candidates(image_path)

    assert [fact["bbox"] for fact in facts] == [[10, 10, 20, 20], [50, 20, 30, 20]]
    assert [fact["value"] for fact in facts] == ["55", "42"]
    assert [fact["source"] for fact in facts] == ["fine_tuned", "fine_tuned"]
    assert facts[0]["confidence"] == pytest.approx(0.9)
    assert facts[0]["score"] == pytest.approx(0.8)
    assert facts[1]["confidence"] == pytest.approx(0.8)
    assert facts[1]["score"] == pytest.approx(0.9)


def test_extract_numeric_bbox_candidates_scales_only_fine_tuned_hyphen_boxes(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def _fake_detector(_images):
        return [
            {
                "words": np.array(
                    [
                        [0.10, 0.10, 0.30, 0.30, 0.8],
                        [0.50, 0.20, 0.80, 0.40, 0.9],
                    ],
                    dtype=np.float32,
                )
            }
        ]

    monkeypatch.setattr(local_doctr, "_load_detector_for_backend", lambda _backend: _fake_detector)
    monkeypatch.setattr(
        local_doctr,
        "_recognize_crop_payloads",
        lambda crops, cancel_requested=None: [("-", 0.9), ("42", 0.8)],
    )

    facts = extract_numeric_bbox_candidates(image_path)

    assert [fact["bbox"] for fact in facts] == [[16, 16, 7, 7], [50, 20, 30, 20]]
    assert [fact["value"] for fact in facts] == ["-", "42"]


def test_extract_numeric_bbox_candidates_filters_excluded_values_and_keeps_empty_text(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    def _fake_detector(_images):
        return [
            {
                "words": np.array(
                    [
                        [0.05, 0.10, 0.20, 0.30, 0.9],
                        [0.25, 0.10, 0.40, 0.30, 0.9],
                        [0.45, 0.10, 0.60, 0.30, 0.9],
                        [0.65, 0.10, 0.80, 0.30, 0.9],
                        [0.05, 0.40, 0.20, 0.60, 0.9],
                    ],
                    dtype=np.float32,
                )
            }
        ]

    monkeypatch.setattr(local_doctr, "_load_detector_for_backend", lambda _backend: _fake_detector)
    monkeypatch.setattr(
        local_doctr,
        "_recognize_crop_payloads",
        lambda crops, cancel_requested=None: [("", None), ("31", 0.8), ("2026", 0.8), ("310", 0.8), ("notes", 0.8)],
    )

    facts = extract_numeric_bbox_candidates(image_path)

    assert [fact["bbox"] for fact in facts] == [[6, 8, 18, 16], [77, 8, 19, 16]]
    assert [fact["value"] for fact in facts] == ["", "310"]
    assert [fact["source"] for fact in facts] == ["fine_tuned", "fine_tuned"]
    assert facts[0]["confidence"] is None
    assert facts[1]["confidence"] == pytest.approx(0.8)


def test_extract_numeric_bbox_facts_honors_max_facts_for_single_backend(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    monkeypatch.setattr(
        local_doctr,
        "extract_numeric_bbox_candidates",
        lambda *args, **kwargs: [
            {"bbox": [10, 10, 10, 10], "value": "1", "source": "fine_tuned", "score": 0.9, "confidence": 0.9},
            {"bbox": [30, 10, 10, 10], "value": "2", "source": "fine_tuned", "score": 0.8, "confidence": 0.8},
            {"bbox": [50, 10, 10, 10], "value": "3", "source": "fine_tuned", "score": 0.7, "confidence": 0.7},
        ],
    )

    facts = extract_numeric_bbox_facts(image_path, max_facts=2, backend=LOCAL_DETECTOR_BACKEND_FINE_TUNED)

    assert facts == [
        {"bbox": [10, 10, 10, 10], "value": "1"},
        {"bbox": [30, 10, 10, 10], "value": "2"},
    ]


def test_extract_numeric_bbox_candidates_passes_backend_through_loader(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    captured: dict[str, object] = {}

    def _fake_loader(backend):
        captured["backend"] = backend
        return lambda _images: [{"words": np.array([[0.10, 0.10, 0.20, 0.20, 0.9]], dtype=np.float32)}]

    monkeypatch.setattr(local_doctr, "_load_detector_for_backend", _fake_loader)
    monkeypatch.setattr(local_doctr, "_recognize_crop_payloads", lambda crops, cancel_requested=None: [("7", 0.7)])

    facts = extract_numeric_bbox_candidates(image_path, backend=LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED)

    assert captured["backend"] == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED
    assert facts == [{"bbox": [10, 10, 10, 10], "confidence": 0.7, "score": pytest.approx(0.9), "source": "old", "value": "7"}]


def test_extract_numeric_bbox_facts_merged_uses_old_geometry_and_fine_value(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def _fake_extract(_image_path, *, cancel_requested=None, backend):
        if backend == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
            return [{"bbox": [10, 10, 10, 10], "value": "100", "source": "old", "score": 0.6, "confidence": 0.6}]
        return [{"bbox": [8, 8, 20, 20], "value": "-", "source": "fine_tuned", "score": 0.9, "confidence": 0.9}]

    monkeypatch.setattr(local_doctr, "extract_numeric_bbox_candidates", _fake_extract)

    facts = extract_numeric_bbox_facts(image_path, backend=LOCAL_DETECTOR_BACKEND_MERGED)

    assert facts == [{"bbox": [10, 10, 10, 10], "value": "-"}]


def test_extract_numeric_bbox_facts_merged_drops_unmatched_old_and_keeps_unmatched_fine(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def _fake_extract(_image_path, *, cancel_requested=None, backend):
        if backend == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
            return [{"bbox": [10, 10, 10, 10], "value": "10", "source": "old", "score": 0.6, "confidence": 0.6}]
        return [{"bbox": [60, 60, 10, 10], "value": "-", "source": "fine_tuned", "score": 0.9, "confidence": 0.9}]

    monkeypatch.setattr(local_doctr, "extract_numeric_bbox_candidates", _fake_extract)

    facts = extract_numeric_bbox_facts(image_path, backend=LOCAL_DETECTOR_BACKEND_MERGED)

    assert facts == [{"bbox": [60, 60, 10, 10], "value": "-"}]


def test_extract_numeric_bbox_facts_merged_matches_low_iou_high_containment(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def _fake_extract(_image_path, *, cancel_requested=None, backend):
        if backend == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
            return [{"bbox": [10, 10, 10, 10], "value": "55", "source": "old", "score": 0.6, "confidence": 0.8}]
        return [{"bbox": [8, 8, 30, 30], "value": "55", "source": "fine_tuned", "score": 0.9, "confidence": 0.9}]

    monkeypatch.setattr(local_doctr, "extract_numeric_bbox_candidates", _fake_extract)

    facts = extract_numeric_bbox_facts(image_path, backend=LOCAL_DETECTOR_BACKEND_MERGED)

    assert facts == [{"bbox": [10, 10, 10, 10], "value": "55"}]


def test_extract_numeric_bbox_facts_writes_debug_log(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    logged_payloads: list[dict[str, object]] = []

    def _fake_extract(_image_path, *, cancel_requested=None, backend):
        if backend == LOCAL_DETECTOR_BACKEND_DOCTR_PRETRAINED:
            return [{"bbox": [10, 10, 10, 10], "value": "10", "source": "old", "score": 0.6, "confidence": 0.6}]
        return [{"bbox": [8, 8, 20, 20], "value": "10", "source": "fine_tuned", "score": 0.9, "confidence": 0.9}]

    monkeypatch.setattr(local_doctr, "extract_numeric_bbox_candidates", _fake_extract)
    monkeypatch.setattr(
        local_doctr,
        "_write_doctr_debug_log",
        lambda _image_path, payload: logged_payloads.append(payload) or (tmp_path / "debug.json"),
    )

    facts = extract_numeric_bbox_facts(image_path, backend=LOCAL_DETECTOR_BACKEND_MERGED)

    assert facts == [{"bbox": [10, 10, 10, 10], "value": "10"}]
    assert len(logged_payloads) == 1
    assert logged_payloads[0]["backend"] == LOCAL_DETECTOR_BACKEND_MERGED
    assert logged_payloads[0]["facts"] == facts
    assert logged_payloads[0]["old_candidates"][0]["bbox"] == [10, 10, 10, 10]
    assert logged_payloads[0]["fine_candidates"][0]["bbox"] == [8, 8, 20, 20]
