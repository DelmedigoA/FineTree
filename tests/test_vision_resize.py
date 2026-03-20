from __future__ import annotations

import sys
import types

import finetree_annotator.vision_resize as resize_mod


def _install_fake_qwen_vl_utils(monkeypatch, *, height: int, width: int, require_factor: bool = False) -> dict[str, object]:
    calls: dict[str, object] = {}
    vision_mod = types.ModuleType("qwen_vl_utils.vision_process")

    if require_factor:
        def _smart_resize(orig_height: int, orig_width: int, factor: int, min_pixels=None, max_pixels=None):
            calls["orig_height"] = orig_height
            calls["orig_width"] = orig_width
            calls["factor"] = factor
            calls["min_pixels"] = min_pixels
            calls["max_pixels"] = max_pixels
            return height, width
    else:
        def _smart_resize(orig_height: int, orig_width: int, **kwargs):
            calls["orig_height"] = orig_height
            calls["orig_width"] = orig_width
            calls["kwargs"] = dict(kwargs)
            return height, width

    vision_mod.smart_resize = _smart_resize
    root_mod = types.ModuleType("qwen_vl_utils")
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", root_mod)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_mod)
    return calls


def test_smart_resize_dimensions_uses_qwen_when_available(monkeypatch) -> None:
    calls = _install_fake_qwen_vl_utils(monkeypatch, height=336, width=672)

    out = resize_mod.smart_resize_dimensions(500, 1000, min_pixels=200_000, max_pixels=300_000)

    assert out == (336, 672)
    assert calls == {
        "orig_height": 500,
        "orig_width": 1000,
        "kwargs": {"min_pixels": 200_000, "max_pixels": 300_000},
    }


def test_smart_resize_dimensions_supports_required_factor_signature(monkeypatch) -> None:
    calls = _install_fake_qwen_vl_utils(monkeypatch, height=120, width=240, require_factor=True)

    out = resize_mod.smart_resize_dimensions(500, 1000, min_pixels=200_000, max_pixels=300_000)

    assert out == (120, 240)
    assert calls["factor"] == resize_mod.DEFAULT_QWEN_VISION_FACTOR


def test_smart_resize_dimensions_falls_back_when_qwen_import_fails(monkeypatch) -> None:
    original_import = __import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "qwen_vl_utils.vision_process":
            raise ModuleNotFoundError("No module named 'torchvision'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    h, w = resize_mod.smart_resize_dimensions(2301, 1657, min_pixels=None, max_pixels=1_200_000)

    assert h % resize_mod.DEFAULT_QWEN_VISION_FACTOR == 0
    assert w % resize_mod.DEFAULT_QWEN_VISION_FACTOR == 0
    assert h * w <= 1_200_000


def test_restore_bbox_from_resized_pixels_scales_back_to_original(monkeypatch) -> None:
    monkeypatch.setattr(resize_mod, "smart_resize_dimensions", lambda *_args, **_kwargs: (250, 500))

    restored = resize_mod.restore_bbox_from_resized_pixels(
        {"x": 50, "y": 25, "w": 100, "h": 50},
        original_width=1000,
        original_height=500,
        max_pixels=100_000,
    )

    assert restored == {"x": 100.0, "y": 50.0, "w": 200.0, "h": 100.0}


def test_restore_bbox_from_resized_pixels_is_noop_under_budget(monkeypatch) -> None:
    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("smart_resize_dimensions should not run when image is already under max_pixels")

    monkeypatch.setattr(resize_mod, "smart_resize_dimensions", _should_not_run)

    restored = resize_mod.restore_bbox_from_resized_pixels(
        {"x": 10, "y": 20, "w": 30, "h": 40},
        original_width=1000,
        original_height=500,
        max_pixels=600_000,
    )

    assert restored == {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0}


def test_restore_bbox_from_resized_pixels_clamps_to_original_bounds(monkeypatch) -> None:
    monkeypatch.setattr(resize_mod, "smart_resize_dimensions", lambda *_args, **_kwargs: (250, 500))

    restored = resize_mod.restore_bbox_from_resized_pixels(
        {"x": 480, "y": 240, "w": 100, "h": 30},
        original_width=1000,
        original_height=500,
        max_pixels=100_000,
    )

    assert restored == {"x": 960.0, "y": 480.0, "w": 40.0, "h": 20.0}


def test_prepared_dimensions_for_max_pixels_returns_original_when_under_budget(monkeypatch) -> None:
    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("smart_resize_dimensions should not run when image is already under max_pixels")

    monkeypatch.setattr(resize_mod, "smart_resize_dimensions", _should_not_run)

    out = resize_mod.prepared_dimensions_for_max_pixels(
        original_width=1000,
        original_height=500,
        max_pixels=600_000,
    )

    assert out == (500, 1000)
