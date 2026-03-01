from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.finetune.push_dataset_hub import export_for_hf


def test_export_for_hf_rewrites_image_paths_and_copies_images(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    image_path.write_bytes(b"img")

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": f"Prompt for {image_path}"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
        "metadata": {"document_id": "doc1"},
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export"
    train_rows, val_rows = export_for_hf(root, export_dir)

    assert train_rows == 1
    assert val_rows == 0
    exported_train = (export_dir / "train.jsonl").read_text(encoding="utf-8").strip()
    out = json.loads(exported_train)
    rel_img = out["messages"][0]["content"][0]["image"]
    assert rel_img == "images/doc1/page_0001.png"
    assert (export_dir / rel_img).is_file()
