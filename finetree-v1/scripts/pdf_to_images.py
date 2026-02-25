from pathlib import Path
import argparse

from pdf2image import convert_from_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF file to page images.")
    parser.add_argument("pdf_path", help="Path to PDF inside data/raw/")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Invalid PDF path: {pdf_path}")

    doc_id = pdf_path.stem
    output_dir = Path("data/pages") / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path))
    for i, image in enumerate(images, start=1):
        output_path = output_dir / f"page_{i:04d}.png"
        image.save(output_path, "PNG")

    print(f"Saved {len(images)} pages to {output_dir}")


if __name__ == "__main__":
    main()
