from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
from finetree_annotator.ai.types import LocalDetectorBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
WET_IMAGE_PATH = (
    REPO_ROOT / "artifacts" / "hf_bbox_inspection" / "max_pixels_1000000" / "01_2014_page_0004" / "resized" / "page_0004.png"
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.mark.skipif(
    os.environ.get("FINETREE_RUN_DETECTOR_WET_TEST") != "1",
    reason="Set FINETREE_RUN_DETECTOR_WET_TEST=1 to run the local detector wet test.",
)
@pytest.mark.skipif(not _module_available("doctr"), reason="python-doctr is not installed.")
@pytest.mark.skipif(not _module_available("PIL"), reason="Pillow is not installed.")
def test_local_doctr_wet_test_writes_overlay_and_json(tmp_path: Path) -> None:
    from scripts.run_local_doctr_wet_test import run_wet_test

    assert WET_IMAGE_PATH.is_file()
    json_path, overlay_path, fact_count = run_wet_test(
        image_path=WET_IMAGE_PATH,
        output_dir=tmp_path,
        backend=LocalDetectorBackend.FINE_TUNED.value,
    )

    assert json_path.is_file()
    assert overlay_path.is_file()
    assert fact_count > 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload.get("backend") == LocalDetectorBackend.FINE_TUNED.value
    values = [str(fact.get("value", "")).strip() for fact in payload.get("facts", [])]
    assert "31" not in values
    excluded_years = {str(year) for year in range(2000, 2027)}
    assert not any(value in excluded_years for value in values)
