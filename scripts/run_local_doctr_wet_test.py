from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.local_doctr import extract_numeric_bbox_facts  # noqa: E402
from finetree_annotator.ai.types import LocalDetectorBackend  # noqa: E402


DEFAULT_IMAGE_PATH = (
    REPO_ROOT / "artifacts" / "hf_bbox_inspection" / "max_pixels_1000000" / "01_2014_page_0004" / "resized" / "page_0004.png"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "detector_wet_test"


def run_wet_test(
    *,
    image_path: Path,
    output_dir: Path,
    max_facts: int = 0,
    backend: str = LocalDetectorBackend.MERGED.value,
) -> tuple[Path, Path, int]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required to write the wet-test overlay image. Install project dependencies first."
        ) from exc

    image_path = image_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    facts = extract_numeric_bbox_facts(image_path, max_facts=max_facts, backend=backend)
    json_path = output_dir / f"{image_path.stem}.detected_bboxes.json"
    overlay_path = output_dir / f"{image_path.stem}.overlay.png"

    json_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "backend": str(backend),
                "fact_count": len(facts),
                "facts": facts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        for fact in facts:
            bbox = fact.get("bbox") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            x, y, w, h = [int(v) for v in bbox[:4]]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        img.save(overlay_path)

    return json_path, overlay_path, len(facts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local detector backends on a repo image and save bbox outputs.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"Image path to detect. Default: {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON + overlay outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-facts",
        type=int,
        default=0,
        help="Optional cap on emitted detected facts. 0 means no cap.",
    )
    parser.add_argument(
        "--backend",
        choices=(
            LocalDetectorBackend.MERGED.value,
            LocalDetectorBackend.FINE_TUNED.value,
            LocalDetectorBackend.DOCTR_PRETRAINED.value,
        ),
        default=LocalDetectorBackend.MERGED.value,
        help="Detector backend to run. Default: merged",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, overlay_path, fact_count = run_wet_test(
        image_path=args.image,
        output_dir=args.output_dir,
        max_facts=max(0, int(args.max_facts)),
        backend=str(args.backend),
    )
    print(f"Image: {args.image.resolve()}")
    print(f"Backend: {args.backend}")
    print(f"Detected facts: {fact_count}")
    print(f"JSON: {json_path}")
    print(f"Overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
