from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.finetune.push_dataset_hub_no_bbox import export_for_hf_no_bbox


def test_export_for_hf_no_bbox_strips_bbox_from_text_payload(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    image_path.write_bytes(b"img")

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    target_payload = {
        "meta": {"type": "other"},
        "facts": [
            {
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "value": "10",
                "refference": "",
                "path": ["a"],
            }
        ],
    }
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": f"Prompt for {image_path}"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target_payload)}]},
        ],
        "metadata": {"document_id": "doc1"},
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export_no_bbox"
    train_rows, val_rows = export_for_hf_no_bbox(root, export_dir)

    assert train_rows == 1
    assert val_rows == 0
    exported_train = (export_dir / "train.jsonl").read_text(encoding="utf-8").strip()
    out = json.loads(exported_train)
    rel_img = out["image"]
    assert rel_img == "images/doc1/page_0001.png"
    assert "Prompt for" in out["instruction"]
    assert (export_dir / rel_img).is_file()

    output_payload = json.loads(out["text"])
    assert "bbox" not in output_payload["facts"][0]
    assert output_payload["facts"][0]["value"] == "10"
