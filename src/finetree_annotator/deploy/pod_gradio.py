from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from ..qwen_vlm import generate_content_from_image, stream_content_from_image


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_example_image_path() -> Optional[Path]:
    explicit = str(os.getenv("FINETREE_GRADIO_EXAMPLE_IMAGE") or "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.is_file():
            return explicit_path.resolve()

    root = _project_root()
    preferred = root / "data" / "pdf_images" / "test" / "page_0001.png"
    if preferred.is_file():
        return preferred.resolve()

    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = sorted((root / "data" / "pdf_images").rglob(pattern))
        for match in matches:
            if match.is_file():
                return match.resolve()
    return None


def _infer_stream(
    image_obj,
    prompt: str,
    model: str,
    config_path: Optional[str],
) -> Iterator[str]:
    if image_obj is None:
        yield "Please upload an image."
        return

    prompt_text = str(prompt or "").strip() or "What is shown in this page? Keep it short."
    model_name = str(model or "").strip() or None

    with tempfile.TemporaryDirectory(prefix="finetree-pod-gradio-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        image_path = temp_dir / "input.png"
        image_obj.save(image_path)
        acc = ""
        for chunk in stream_content_from_image(
            image_path=image_path,
            prompt=prompt_text,
            model=model_name,
            config_path=config_path,
        ):
            if not chunk:
                continue
            acc += chunk
            yield acc


def _infer_once(image_obj, prompt: str, model: str, config_path: Optional[str]) -> str:
    if image_obj is None:
        return "Please upload an image."
    prompt_text = str(prompt or "").strip() or "What is shown in this page? Keep it short."
    model_name = str(model or "").strip() or None
    with tempfile.TemporaryDirectory(prefix="finetree-pod-gradio-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        image_path = temp_dir / "input.png"
        image_obj.save(image_path)
        return generate_content_from_image(
            image_path=image_path,
            prompt=prompt_text,
            model=model_name,
            config_path=config_path,
        )


def _require_basic_auth_credentials() -> tuple[str, str]:
    username = str(os.getenv("FINETREE_GRADIO_USER") or "").strip()
    password = str(os.getenv("FINETREE_GRADIO_PASS") or "").strip()
    if not username or not password:
        raise RuntimeError(
            "Gradio auth is required. Set FINETREE_GRADIO_USER and FINETREE_GRADIO_PASS."
        )
    return username, password


def build_app(config_path: Optional[str] = None):
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pod Gradio app requires gradio. Install with `pip install gradio`.") from exc

    example_path = _default_example_image_path()
    default_prompt = "Extract page JSON using the FineTree schema."

    with gr.Blocks(title="FineTree Qwen Pod Tester") as app:
        gr.Markdown("## FineTree Qwen Test UI (Pod Port 5555)")
        if example_path is not None:
            gr.Markdown(f"Loaded example image: `{example_path}`")
        else:
            gr.Markdown("No default example image found under `data/pdf_images`.")
        with gr.Row():
            image = gr.Image(
                type="pil",
                label="Page image",
                value=str(example_path) if example_path is not None else None,
            )
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value=default_prompt,
                lines=2,
            )
        with gr.Row():
            model = gr.Textbox(label="Model override (optional)", value="")
        with gr.Row():
            stream = gr.Checkbox(label="Stream output", value=True)
        output = gr.Textbox(label="Output", lines=14)
        run = gr.Button("Run inference")

        def _run(image_obj, prompt_text, model_name, stream_enabled):
            if bool(stream_enabled):
                return _infer_stream(image_obj, prompt_text, model_name, config_path)
            return _infer_once(image_obj, prompt_text, model_name, config_path)

        run.click(_run, inputs=[image, prompt, model, stream], outputs=output)

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineTree Pod Gradio test app.")
    parser.add_argument("--config", default=None, help="FineTree YAML config path.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5555, help="Bind port (default: 5555).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    username, password = _require_basic_auth_credentials()
    app = build_app(config_path=args.config)
    app.queue()
    app.launch(
        server_name=str(args.host),
        server_port=int(args.port),
        show_error=True,
        auth=(username, password),
        prevent_thread_lock=True,
    )
    try:
        import time

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
