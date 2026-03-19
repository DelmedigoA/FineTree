from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import workspace as workspace_mod


@dataclass(frozen=True)
class StartupContext:
    mode: str
    images_dir: Optional[Path] = None
    annotations_path: Optional[Path] = None
    pdf_path: Optional[Path] = None


def default_annotations_path(images_dir: Path) -> Path:
    return workspace_mod.default_annotations_path(images_dir)


def resolve_startup_context(
    input_path_arg: Optional[str],
    images_dir_arg: Optional[str],
    annotations_arg: Optional[str],
) -> StartupContext:
    if input_path_arg and images_dir_arg:
        raise ValueError("Cannot provide both positional input_path and --images-dir.")

    if input_path_arg:
        input_path = Path(input_path_arg).expanduser().resolve()
        if input_path.is_dir():
            annotations_path = Path(annotations_arg) if annotations_arg else default_annotations_path(input_path)
            return StartupContext(mode="images-dir", images_dir=input_path, annotations_path=annotations_path)
        if not input_path.exists():
            raise ValueError(f"Input path not found: {input_path}")
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            images_dir = (Path("data/pdf_images") / input_path.stem).resolve()
            annotations_path = Path(annotations_arg) if annotations_arg else default_annotations_path(images_dir)
            return StartupContext(
                mode="pdf",
                images_dir=images_dir,
                annotations_path=annotations_path,
                pdf_path=input_path,
            )
        raise ValueError(f"Unsupported input path: {input_path}. Provide a PDF file or an image directory.")

    if images_dir_arg:
        images_dir = Path(images_dir_arg).expanduser().resolve()
        annotations_path = Path(annotations_arg) if annotations_arg else default_annotations_path(images_dir)
        return StartupContext(mode="images-dir", images_dir=images_dir, annotations_path=annotations_path)

    return StartupContext(mode="home")


__all__ = ["StartupContext", "default_annotations_path", "resolve_startup_context"]
