from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.gemini_few_shot import (
    DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
    DEFAULT_TEST_FEW_SHOT_PAGES,
    load_complex_few_shot_examples,
    load_test_pdf_few_shot_examples,
    resolve_repo_relative_path,
)


def _write_test_annotation(path: Path, page_names: list[str]) -> None:
    pages = []
    for page_name in page_names:
        pages.append(
            {
                "image": page_name,
                "meta": {"type": "other", "page_num": None, "entity_name": None, "title": None},
                "facts": [{"value": "10", "refference": "", "path": []}],
            }
        )
    payload = {"images_dir": "data/pdf_images/test", "pages": pages}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_resolve_repo_relative_path_prefers_existing_candidate(tmp_path: Path) -> None:
    target = tmp_path / "data/annotations/test.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    resolved = resolve_repo_relative_path("data/annotations/test.json", repo_roots=[tmp_path])
    assert resolved == target.resolve()


def test_load_test_pdf_few_shot_examples_reads_all_fixed_pages(tmp_path: Path) -> None:
    image_dir = tmp_path / "data/pdf_images/test"
    image_dir.mkdir(parents=True)
    for page_name in DEFAULT_TEST_FEW_SHOT_PAGES:
        (image_dir / page_name).write_bytes(b"img")

    ann_path = tmp_path / "data/annotations/test.json"
    ann_path.parent.mkdir(parents=True)
    _write_test_annotation(ann_path, list(DEFAULT_TEST_FEW_SHOT_PAGES))

    examples, warnings = load_test_pdf_few_shot_examples(repo_roots=[tmp_path])

    assert warnings == []
    assert [Path(ex["image_path"]).name for ex in examples] == list(DEFAULT_TEST_FEW_SHOT_PAGES)
    for ex in examples:
        parsed = json.loads(ex["expected_json"])
        assert "meta" in parsed
        assert isinstance(parsed.get("facts"), list)


def test_load_test_pdf_few_shot_examples_skips_missing_with_warning(tmp_path: Path) -> None:
    image_dir = tmp_path / "data/pdf_images/test"
    image_dir.mkdir(parents=True)
    # Only one image + one page in annotations.
    only_page = DEFAULT_TEST_FEW_SHOT_PAGES[0]
    (image_dir / only_page).write_bytes(b"img")

    ann_path = tmp_path / "data/annotations/test.json"
    ann_path.parent.mkdir(parents=True)
    _write_test_annotation(ann_path, [only_page])

    examples, warnings = load_test_pdf_few_shot_examples(repo_roots=[tmp_path])

    assert len(examples) == 1
    assert Path(examples[0]["image_path"]).name == only_page
    assert len(warnings) >= 1


def test_load_complex_few_shot_examples_reads_multi_dataset_selection(tmp_path: Path) -> None:
    test_image_dir = tmp_path / "data/pdf_images/test"
    pdf3_image_dir = tmp_path / "data/pdf_images/pdf_3"
    test_image_dir.mkdir(parents=True)
    pdf3_image_dir.mkdir(parents=True)
    (test_image_dir / "page_0009.png").write_bytes(b"img-test")
    (pdf3_image_dir / "page_0018.png").write_bytes(b"img-pdf3")

    test_payload = {
        "images_dir": "data/pdf_images/test",
        "pages": [
            {
                "image": "page_0009.png",
                "meta": {"type": "notes"},
                "facts": [{"value": "10", "refference": "", "path": []}],
            }
        ],
    }
    pdf3_payload = {
        "images_dir": "data/pdf_images/pdf_3",
        "pages": [
            {
                "image": "page_0018.png",
                "meta": {"type": "notes"},
                "facts": [{"value": "20", "refference": "", "path": []}],
            }
        ],
    }

    (tmp_path / "data/annotations").mkdir(parents=True)
    (tmp_path / "data/annotations/test.json").write_text(json.dumps(test_payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "data/annotations/pdf_3.json").write_text(json.dumps(pdf3_payload, ensure_ascii=False), encoding="utf-8")

    examples, warnings = load_complex_few_shot_examples(
        repo_roots=[tmp_path],
        selections=(("test", "page_0009.png"), ("pdf_3", "page_0018.png")),
    )

    assert warnings == []
    assert len(examples) == 2
    assert [Path(ex["image_path"]).name for ex in examples] == ["page_0009.png", "page_0018.png"]


def test_default_complex_few_shot_selection_has_seven_items() -> None:
    assert len(DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS) == 7
