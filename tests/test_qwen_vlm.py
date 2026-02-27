from __future__ import annotations

from pathlib import Path

from finetree_annotator import qwen_vlm


def test_resolve_config_path_uses_env(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("run: {}\n", encoding="utf-8")
    monkeypatch.setenv("FINETREE_QWEN_CONFIG", str(cfg))

    resolved = qwen_vlm._resolve_config_path(None)
    assert resolved == cfg.resolve()


def test_generate_page_extraction_parses_output(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"x")

    payload = (
        '{"meta":{"entity_name":null,"page_num":null,"type":"other","title":null},'
        '"facts":[{"bbox":{"x":1,"y":2,"w":3,"h":4},"value":"10",'
        '"refference":"","date":null,"path":[],"currency":null,"scale":null,"value_type":"amount"}]}'
    )

    monkeypatch.setattr(qwen_vlm, "generate_content_from_image", lambda **_: payload)
    extraction = qwen_vlm.generate_page_extraction_from_image(image_path=image_path, prompt="p")

    assert extraction.meta.type.value == "other"
    assert len(extraction.facts) == 1
    assert extraction.facts[0].bbox.x == 1
