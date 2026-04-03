from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.annotation_core import BoxRecord, PageState, default_page_meta
from finetree_annotator.gemini_few_shot import (
    DEFAULT_2015_TWO_SHOT_SELECTIONS,
    DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
    DEFAULT_TEST_ONE_SHOT_PAGE,
    DEFAULT_TEST_FEW_SHOT_PAGES,
    load_complex_few_shot_examples,
    load_document_few_shot_examples,
    load_test_pdf_few_shot_examples,
    parse_document_few_shot_page_spec,
    resolve_repo_relative_path,
)
from finetree_annotator.schema_contract import PROMPT_FACT_KEYS, PROMPT_PAGE_META_KEYS


def _write_test_annotation(path: Path, page_names: list[str]) -> None:
    pages = []
    for page_name in page_names:
        pages.append(
            {
                "image": page_name,
                "meta": {
                    "entity_name": None,
                    "page_num": None,
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                    "annotation_note": "runtime note",
                    "annotation_status": "approved",
                },
                "facts": [
                    {
                        "bbox": [1, 2, 3, 4],
                        "value": "10",
                        "fact_num": 1,
                        "equations": None,
                        "natural_sign": "positive",
                        "row_role": "detail",
                        "comment_ref": None,
                        "note_flag": False,
                        "note_name": None,
                        "note_num": None,
                        "note_ref": None,
                        "date": "2024-12-31",
                        "period_type": None,
                        "period_start": None,
                        "period_end": None,
                        "duration_type": None,
                        "recurring_period": None,
                        "path": [],
                        "path_source": None,
                        "currency": None,
                        "scale": None,
                        "value_type": None,
                        "value_context": None,
                    }
                ],
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
        assert "annotation_note" not in parsed["pages"][0]["meta"]
        assert "annotation_status" not in parsed["pages"][0]["meta"]
        assert "date" not in parsed["pages"][0]["facts"][0]
        assert parsed["pages"][0]["facts"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_load_test_pdf_few_shot_examples_uses_exact_live_prompt_schema_keys(tmp_path: Path) -> None:
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
    parsed = json.loads(examples[0]["expected_json"])
    page_payload = parsed["pages"][0]

    assert list(page_payload["meta"].keys()) == list(PROMPT_PAGE_META_KEYS)
    assert list(page_payload["facts"][0].keys()) == ["bbox", *PROMPT_FACT_KEYS]
    assert "annotation_note" not in page_payload["meta"]
    assert "annotation_status" not in page_payload["meta"]
    assert "date" not in page_payload["facts"][0]


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
                "meta": {"page_type": "statements", "statement_type": None},
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


def _page_state(*, approved: bool, value: str) -> PageState:
    meta = default_page_meta()
    meta["annotation_status"] = "approved" if approved else None
    return PageState(
        meta=meta,
        facts=[BoxRecord(bbox={"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}, fact={"value": value, "path": []})],
    )


def test_parse_document_few_shot_page_spec_accepts_single_page_and_range() -> None:
    page_indices, error = parse_document_few_shot_page_spec("3", page_count=6, current_page_index=5)
    assert error is None
    assert page_indices == [2]

    page_indices, error = parse_document_few_shot_page_spec("1-3", page_count=6, current_page_index=5)
    assert error is None
    assert page_indices == [0, 1, 2]


def test_parse_document_few_shot_page_spec_rejects_invalid_inputs() -> None:
    _, error = parse_document_few_shot_page_spec("0", page_count=6, current_page_index=5)
    assert error == "Custom pages must be 1-based page numbers greater than 0."

    _, error = parse_document_few_shot_page_spec("3-1", page_count=6, current_page_index=5)
    assert error == "Custom page ranges must be ascending, for example `1-3`."

    _, error = parse_document_few_shot_page_spec("a-b", page_count=6, current_page_index=5)
    assert error == "Custom pages must be a single page like `3` or a range like `1-3`."

    _, error = parse_document_few_shot_page_spec("8", page_count=6, current_page_index=5)
    assert error == "Custom pages must be within this document's range: 1-6."

    _, error = parse_document_few_shot_page_spec("5-6", page_count=6, current_page_index=4)
    assert error == "Custom few-shot pages cannot include the current page (5)."


def test_load_document_few_shot_examples_selects_nearest_approved_previous_pages(tmp_path: Path) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    page_images = []
    for page_num in range(1, 6):
        image_path = images_dir / f"page_{page_num:04d}.png"
        image_path.write_bytes(b"img")
        page_images.append(image_path)

    page_states = {
        "page_0001.png": _page_state(approved=True, value="10"),
        "page_0002.png": _page_state(approved=False, value="20"),
        "page_0003.png": _page_state(approved=True, value="30"),
        "page_0004.png": _page_state(approved=False, value="40"),
    }

    examples, warnings, error = load_document_few_shot_examples(
        images_dir=images_dir,
        page_images=page_images,
        page_states=page_states,
        current_page_index=4,
        source="previous_pages",
        previous_count=2,
    )

    assert error is None
    assert warnings == []
    assert [Path(example["image_path"]).name for example in examples] == ["page_0001.png", "page_0003.png"]


def test_load_document_few_shot_examples_custom_range_uses_approved_pages_only(tmp_path: Path) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    page_images = []
    for page_num in range(1, 6):
        image_path = images_dir / f"page_{page_num:04d}.png"
        image_path.write_bytes(b"img")
        page_images.append(image_path)

    page_states = {
        "page_0001.png": _page_state(approved=True, value="10"),
        "page_0002.png": _page_state(approved=False, value="20"),
        "page_0003.png": _page_state(approved=True, value="30"),
        "page_0004.png": _page_state(approved=False, value="40"),
    }

    examples, warnings, error = load_document_few_shot_examples(
        images_dir=images_dir,
        page_images=page_images,
        page_states=page_states,
        current_page_index=4,
        source="custom_pages",
        page_spec="1-4",
    )

    assert error is None
    assert warnings == ["Skipped unapproved custom page(s): 2, 4."]
    assert [Path(example["image_path"]).name for example in examples] == ["page_0001.png", "page_0003.png"]


def test_load_document_few_shot_examples_returns_error_when_no_approved_previous_pages_exist(tmp_path: Path) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    page_images = []
    for page_num in range(1, 4):
        image_path = images_dir / f"page_{page_num:04d}.png"
        image_path.write_bytes(b"img")
        page_images.append(image_path)

    page_states = {
        "page_0001.png": _page_state(approved=False, value="10"),
        "page_0002.png": _page_state(approved=False, value="20"),
    }

    examples, warnings, error = load_document_few_shot_examples(
        images_dir=images_dir,
        page_images=page_images,
        page_states=page_states,
        current_page_index=2,
        source="previous_pages",
        previous_count=2,
    )

    assert examples == []
    assert warnings == []
    assert error == "No approved previous pages are available for few-shot."


def test_load_document_few_shot_examples_warns_when_previous_pages_are_partial(tmp_path: Path) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    page_images = []
    for page_num in range(1, 5):
        image_path = images_dir / f"page_{page_num:04d}.png"
        image_path.write_bytes(b"img")
        page_images.append(image_path)

    page_states = {
        "page_0002.png": _page_state(approved=True, value="20"),
    }

    examples, warnings, error = load_document_few_shot_examples(
        images_dir=images_dir,
        page_images=page_images,
        page_states=page_states,
        current_page_index=3,
        source="previous_pages",
        previous_count=3,
    )

    assert error is None
    assert [Path(example["image_path"]).name for example in examples] == ["page_0002.png"]
    assert warnings == ["Requested 3 previous approved page(s), found 1."]
