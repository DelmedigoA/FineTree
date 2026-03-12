from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.gemini_few_shot import (
    DEFAULT_2015_TWO_SHOT_SELECTIONS,
    DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
    DEFAULT_TEST_ONE_SHOT_PAGE,
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
                "meta": {
                    "page_type": "other",
                    "statement_type": None,
                    "page_num": None,
                    "entity_name": None,
                    "title": None,
                },
                "facts": [{"value": "10", "note_ref": None, "path": []}],
            }
        )
    payload = {"images_dir": "data/pdf_images/test", "metadata": {}, "pages": pages}
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
        assert "metadata" not in parsed
        assert "images_dir" not in parsed
        assert isinstance(parsed.get("pages"), list)
        assert len(parsed["pages"]) == 1
        assert isinstance(parsed["pages"][0].get("facts"), list)


def test_load_test_pdf_few_shot_examples_supports_one_shot_selection(tmp_path: Path) -> None:
    image_dir = tmp_path / "data/pdf_images/test"
    image_dir.mkdir(parents=True)
    (image_dir / DEFAULT_TEST_ONE_SHOT_PAGE).write_bytes(b"img")

    ann_path = tmp_path / "data/annotations/test.json"
    ann_path.parent.mkdir(parents=True)
    _write_test_annotation(ann_path, [DEFAULT_TEST_ONE_SHOT_PAGE])

    examples, warnings = load_test_pdf_few_shot_examples(
        repo_roots=[tmp_path],
        page_names=(DEFAULT_TEST_ONE_SHOT_PAGE,),
    )

    assert warnings == []
    assert len(examples) == 1
    assert Path(examples[0]["image_path"]).name == DEFAULT_TEST_ONE_SHOT_PAGE


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
        "metadata": {},
        "pages": [
            {
                "image": "page_0009.png",
                "meta": {"page_type": "statements", "statement_type": "notes_to_financial_statements"},
                "facts": [{"value": "10", "note_ref": None, "path": []}],
            }
        ],
    }
    pdf3_payload = {
        "images_dir": "data/pdf_images/pdf_3",
        "metadata": {},
        "pages": [
            {
                "image": "page_0018.png",
                "meta": {"page_type": "statements", "statement_type": "notes_to_financial_statements"},
                "facts": [{"value": "20", "note_ref": None, "path": []}],
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


def test_load_complex_few_shot_examples_reads_2015_two_shot_selection(tmp_path: Path) -> None:
    image_dir = tmp_path / "data/pdf_images/2015"
    image_dir.mkdir(parents=True)
    for page_name in ("page_0004.png", "page_0011.png"):
        (image_dir / page_name).write_bytes(b"img")

    payload = {
        "images_dir": "data/pdf_images/2015",
        "metadata": {},
        "pages": [
            {
                "image": "page_0004.png",
                "meta": {"page_type": "statements", "statement_type": "balance_sheet"},
                "facts": [{"value": "10", "note_ref": None, "path": []}],
            },
            {
                "image": "page_0011.png",
                "meta": {"page_type": "statements", "statement_type": "other"},
                "facts": [{"value": "20", "note_ref": None, "path": []}],
            },
        ],
    }

    ann_dir = tmp_path / "data/annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "2015.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    examples, warnings = load_complex_few_shot_examples(
        repo_roots=[tmp_path],
        selections=DEFAULT_2015_TWO_SHOT_SELECTIONS,
    )

    assert warnings == []
    assert [Path(ex["image_path"]).name for ex in examples] == ["page_0004.png", "page_0011.png"]


def test_default_complex_few_shot_selection_has_seven_items() -> None:
    assert len(DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS) == 7
