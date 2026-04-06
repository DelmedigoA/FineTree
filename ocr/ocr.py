#!/usr/bin/env python3
"""
OCR script using LightOnOCR-2 on Apple Silicon (MLX).

Usage:
    python ocr.py                  # opens a file picker
    python ocr.py path/to/image.png
"""

import sys
import argparse
from pathlib import Path

MODEL_ID = "mlx-community/LightOnOCR-2-1B-bf16"


def pick_image_file() -> str | None:
    """Open a native macOS file dialog to pick an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select an image for OCR",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path or None
    except Exception as e:
        print(f"File dialog unavailable: {e}")
        return None


def clean_output(text: str) -> str:
    """Strip chat-template artifacts (system/user/assistant markers)."""
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return text.strip()


def run_ocr(image_path: str, max_tokens: int = 2048) -> str:
    """Load the MLX model and extract text from the image."""
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"Loading model: {MODEL_ID} ...")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)

    # apply_chat_template injects <|image_pad|> so the model actually sees the image
    prompt = apply_chat_template(
        processor,
        config,
        "Extract all text from this image.",
        num_images=1,
    )

    print(f"Running OCR on: {image_path}")
    # generate signature: (model, processor, prompt, image, ...)
    output = generate(
        model,
        processor,
        prompt,
        image_path,
        max_tokens=max_tokens,
        temperature=0.2,
        verbose=False,
    )

    text = output.text if hasattr(output, "text") else str(output)
    return clean_output(text)


def main():
    parser = argparse.ArgumentParser(
        description="OCR an image with LightOnOCR-2 on Apple Silicon"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to image file (omit to open a file picker)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output text file (default: <image_name>.txt next to the image)",
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("No image path provided — opening file picker...")
        image_path = pick_image_file()

    if not image_path:
        print("No image selected. Exiting.")
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    text = run_ocr(image_path, max_tokens=args.max_tokens)

    print("\n--- Extracted Text ---")
    print(text)

    default_out = Path(__file__).parent / "outputs" / (Path(image_path).stem + ".txt")
    out_path = args.output or str(default_out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
