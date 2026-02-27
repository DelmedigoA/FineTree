from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.finetune.config import FinetuneConfig
from finetree_annotator.finetune.preflight import _check_hf_connectivity, run_preflight


def _write_annotation(path: Path, images_dir: Path, image_name: str, with_facts: bool = True) -> None:
    facts = (
        [
            {
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "value": "10",
                "refference": "",
                "date": None,
                "path": ["assets"],
                "currency": "ILS",
                "scale": 1,
                "value_type": "amount",
            }
        ]
        if with_facts
        else []
    )
    payload = {
        "images_dir": str(images_dir),
        "pages": [
            {
                "image": image_name,
                "meta": {"entity_name": "ACME", "page_num": "1", "type": "other", "title": None},
                "facts": facts,
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_preflight_passes_on_valid_local_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ann_dir = tmp_path / "data" / "annotations"
    img_dir = tmp_path / "data" / "pdf_images" / "doc1"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    image_name = "page_0001.png"
    (img_dir / image_name).write_bytes(b"img")
    _write_annotation(ann_dir / "doc1.json", img_dir, image_name, with_facts=True)

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
            },
            "prompt": {
                "use_custom_prompt": False,
            },
        }
    )

    checks = run_preflight(cfg, check_stack=False, check_cuda=False, check_data=True)
    assert all(c.ok for c in checks)


def test_preflight_fails_when_images_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ann_dir = tmp_path / "data" / "annotations"
    img_dir = tmp_path / "data" / "pdf_images" / "doc2"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    _write_annotation(ann_dir / "doc2.json", img_dir, "page_0001.png", with_facts=True)

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
            },
            "prompt": {
                "use_custom_prompt": False,
            },
        }
    )

    checks = run_preflight(cfg, check_stack=False, check_cuda=False, check_data=True)
    assert any((not c.ok and c.name == "annotations") for c in checks)


def test_preflight_fails_when_prompt_required_and_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ann_dir = tmp_path / "data" / "annotations"
    img_dir = tmp_path / "data" / "pdf_images" / "doc3"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    image_name = "page_0001.png"
    (img_dir / image_name).write_bytes(b"img")
    _write_annotation(ann_dir / "doc3.json", img_dir, image_name, with_facts=True)

    cfg = FinetuneConfig.model_validate(
        {
            "data": {
                "annotations_glob": "data/annotations/*.json",
                "images_root": ".",
            },
            "prompt": {
                "use_custom_prompt": True,
                "prompt_path": "missing_prompt.txt",
            },
        }
    )

    checks = run_preflight(cfg, check_stack=False, check_cuda=False, check_data=False)
    assert any((not c.ok and c.name == "config-paths") for c in checks)


def test_hf_check_skips_when_push_disabled() -> None:
    cfg = FinetuneConfig.model_validate({})
    result = _check_hf_connectivity(cfg, force=False)
    assert result.ok is True
    assert result.name == "huggingface"


def test_hf_check_fails_fast_on_invalid_token_prefix() -> None:
    cfg = FinetuneConfig.model_validate(
        {
            "push_to_hub": {
                "enabled": True,
                "repo_id": "org/repo",
                "hf_token": "invalid_token",
            }
        }
    )
    result = _check_hf_connectivity(cfg, force=True)
    assert result.ok is False
    assert "hf_" in result.message
