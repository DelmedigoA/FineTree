"""Minimal CLI that outputs page images to data/pdf_images/<pdf name>."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pdf2image import convert_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump PDF pages into data/pdf_images/<name>/")
    parser.add_argument("pdf", help="Path to the PDF to convert")
    parser.add_argument("--dpi", type=int, default=200, help="Image resolution")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    output_dir = Path("data/pdf_images") / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing images to {output_dir}")

    try:
        pages = convert_from_path(str(pdf_path), dpi=args.dpi, use_pdftocairo=True)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    for i, page in enumerate(pages, start=1):
        target = output_dir / f"page_{i:04d}.png"
        page.save(target, format="PNG")
        print(f"  wrote {target}")

    print(f"Completed {len(pages)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
