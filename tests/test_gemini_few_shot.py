from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.gemini_few_shot import (
    DEFAULT_TEST_FEW_SHOT_PAGES,
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
