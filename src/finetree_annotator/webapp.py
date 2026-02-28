from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .annotation_core import (
    BoxRecord,
    PageState,
    build_annotations_payload,
    default_page_meta,
    load_page_states,
    normalize_bbox_data,
    normalize_fact_data,
    serialize_annotations_json,
)
from .gemini_vlm import parse_page_extraction_text
from .qwen_vlm import generate_content_from_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _discover_page_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    pages = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    pages.sort(key=lambda p: p.name.lower())
    if not pages:
        raise RuntimeError(f"No images found in {images_dir} (expected png/jpg/jpeg/webp).")
    return pages


def _resolve_annotations_path(images_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    return Path("data/annotations") / f"{images_dir.name}.json"


def _serialize_page_state(page_state: PageState) -> dict[str, Any]:
    return {
        "meta": dict(page_state.meta or {}),
        "facts": [
            {
                "bbox": normalize_bbox_data(box.bbox),
                "fact": normalize_fact_data(box.fact),
            }
            for box in page_state.facts
        ],
    }


def _deserialize_page_state(data: dict[str, Any], page_index: int) -> PageState:
    raw_meta = data.get("meta")
    meta = raw_meta if isinstance(raw_meta, dict) else default_page_meta(page_index)
    raw_facts = data.get("facts")
    facts: list[BoxRecord] = []
    if isinstance(raw_facts, list):
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            facts.append(
                BoxRecord(
                    bbox=normalize_bbox_data(item.get("bbox") if isinstance(item.get("bbox"), dict) else None),
                    fact=normalize_fact_data(item.get("fact") if isinstance(item.get("fact"), dict) else None),
                )
            )
    return PageState(meta=meta, facts=facts)


def _load_workspace(images_dir_raw: str, annotations_path_raw: str | None) -> dict[str, Any]:
    images_dir = Path(images_dir_raw).expanduser().resolve()
    page_images = _discover_page_images(images_dir)
    annotations_path = _resolve_annotations_path(images_dir, annotations_path_raw).expanduser().resolve()

    loaded_states: dict[str, PageState] = {}
    if annotations_path.is_file():
        import json

        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        loaded_states = load_page_states(payload, [p.name for p in page_images])

    page_states: dict[str, dict[str, Any]] = {}
    for idx, page in enumerate(page_images):
        state = loaded_states.get(page.name, PageState(meta=default_page_meta(idx), facts=[]))
        page_states[page.name] = _serialize_page_state(state)

    return {
        "images_dir": str(images_dir),
        "annotations_path": str(annotations_path),
        "page_images": [p.name for p in page_images],
        "page_states": page_states,
    }


def _render_prompt(prompt_path_raw: str, image_path: Path) -> str:
    prompt_path = Path(prompt_path_raw).expanduser()
    if not prompt_path.is_file():
        return (
            "Extract page JSON using the FineTree schema. "
            "Return strict JSON with keys: meta, facts.\n"
            f"Image: {image_path.name}"
        )
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.replace("{{PAGE_IMAGE}}", str(image_path))
    prompt = prompt.replace("{{IMAGE_NAME}}", image_path.name)
    return prompt


def _page_json(page_name: str, app_state: dict[str, Any]) -> dict[str, Any]:
    page_data = app_state["page_states"].get(page_name, {"meta": {}, "facts": []})
    return {
        "image": page_name,
        "meta": page_data.get("meta", {}),
        "facts": page_data.get("facts", []),
    }


def _run_qwen_gt(page_name: str, prompt_text: str, app_state: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    if not page_name:
        raise RuntimeError("No page selected.")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise RuntimeError("Prompt is empty.")

    image_path = Path(app_state["images_dir"]) / page_name
    raw_text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt_text,
        config_path=os.getenv("FINETREE_QWEN_CONFIG"),
    )
    parsed = parse_page_extraction_text(raw_text)

    page_facts: list[dict[str, Any]] = []
    for fact in parsed.facts:
        fact_payload = {
            "value": str(fact.value),
            "refference": str(fact.refference),
            "date": fact.date,
            "path": [str(p) for p in fact.path],
            "currency": fact.currency.value if fact.currency is not None else None,
            "scale": int(fact.scale.value) if fact.scale is not None else None,
            "value_type": fact.value_type.value if fact.value_type is not None else None,
        }
        page_facts.append(
            {
                "bbox": normalize_bbox_data(fact.bbox.model_dump(mode="json")),
                "fact": normalize_fact_data(fact_payload),
            }
        )

    app_state["page_states"][page_name] = {
        "meta": parsed.meta.model_dump(mode="json"),
        "facts": page_facts,
    }

    status = f"Qwen GT parsed {len(page_facts)} fact(s) and updated page state."
    return raw_text, parsed.model_dump(mode="json"), _page_json(page_name, app_state), status


def _save_annotations(app_state: dict[str, Any]) -> str:
    images_dir = Path(app_state["images_dir"])
    page_images = [images_dir / name for name in app_state["page_images"]]
    page_states: dict[str, PageState] = {}
    for idx, page_name in enumerate(app_state["page_images"]):
        raw_page_state = app_state["page_states"].get(page_name, {})
        page_states[page_name] = _deserialize_page_state(raw_page_state, idx)

    payload = build_annotations_payload(images_dir=images_dir, page_images=page_images, page_states=page_states)
    annotations_path = Path(app_state["annotations_path"])
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(serialize_annotations_json(payload), encoding="utf-8")
    return f"Saved annotations to {annotations_path}"


def _on_reload(images_dir_raw: str, annotations_path_raw: str, prompt_path_raw: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    app_state = _load_workspace(images_dir_raw, annotations_path_raw if annotations_path_raw.strip() else None)
    first_page = app_state["page_images"][0]
    image_path = str(Path(app_state["images_dir"]) / first_page)
    prompt_text = _render_prompt(prompt_path_raw, Path(image_path))
    return (
        app_state,
        app_state["annotations_path"],
        app_state["images_dir"],
        {"choices": app_state["page_images"], "value": first_page},
        image_path,
        prompt_text,
    )


def _on_page_change(page_name: str, prompt_path_raw: str, app_state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    image_path = str(Path(app_state["images_dir"]) / page_name)
    prompt_text = _render_prompt(prompt_path_raw, Path(image_path))
    return image_path, prompt_text, _page_json(page_name, app_state)


def build_app(images_dir_raw: str, annotations_path_raw: str | None, prompt_path_raw: str):
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gradio is required for web UI. Install with `pip install gradio`.") from exc

    initial_state = _load_workspace(images_dir_raw, annotations_path_raw)
    first_page = initial_state["page_images"][0]
    first_image = str(Path(initial_state["images_dir"]) / first_page)
    first_prompt = _render_prompt(prompt_path_raw, Path(first_image))

    with gr.Blocks(title="FineTree Web UI") as demo:
        gr.Markdown("## FineTree Web UI")
        gr.Markdown("Qwen GT runs in-browser and writes back to FineTree annotation JSON.")

        app_state = gr.State(initial_state)

        with gr.Row():
            images_dir_tb = gr.Textbox(label="Images Directory", value=initial_state["images_dir"])
            annotations_tb = gr.Textbox(label="Annotations JSON", value=initial_state["annotations_path"])
            prompt_path_tb = gr.Textbox(label="Prompt Template Path", value=prompt_path_raw)
            reload_btn = gr.Button("Reload Workspace", variant="secondary")

        page_dd = gr.Dropdown(label="Page", choices=initial_state["page_images"], value=first_page)
        with gr.Row():
            image_view = gr.Image(label="Page Image", value=first_image, type="filepath")
            page_json_view = gr.JSON(label="Current Page State", value=_page_json(first_page, initial_state))

        prompt_tb = gr.Textbox(label="Prompt", value=first_prompt, lines=10)
        run_btn = gr.Button("Run Qwen GT", variant="primary")

        raw_output_tb = gr.Textbox(label="Raw Model Output", lines=12)
        parsed_json_view = gr.JSON(label="Parsed Qwen Output")

        save_btn = gr.Button("Save Annotations File", variant="secondary")
        status_md = gr.Markdown()

        reload_btn.click(
            _on_reload,
            inputs=[images_dir_tb, annotations_tb, prompt_path_tb],
            outputs=[app_state, annotations_tb, images_dir_tb, page_dd, image_view, prompt_tb],
        ).then(
            _on_page_change,
            inputs=[page_dd, prompt_path_tb, app_state],
            outputs=[image_view, prompt_tb, page_json_view],
        )

        page_dd.change(
            _on_page_change,
            inputs=[page_dd, prompt_path_tb, app_state],
            outputs=[image_view, prompt_tb, page_json_view],
        )

        run_btn.click(
            _run_qwen_gt,
            inputs=[page_dd, prompt_tb, app_state],
            outputs=[raw_output_tb, parsed_json_view, page_json_view, status_md],
        )

        save_btn.click(_save_annotations, inputs=[app_state], outputs=[status_md])

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FineTree browser UI with Qwen GT support.")
    parser.add_argument("--images-dir", default="data/pdf_images/test", help="Directory with page images.")
    parser.add_argument("--annotations", default=None, help="Annotations JSON path.")
    parser.add_argument("--prompt-path", default="prompt.txt", help="Prompt template/text file path.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind web server.")
    parser.add_argument("--port", type=int, default=1234, help="Port to bind web server.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = build_app(args.images_dir, args.annotations, args.prompt_path)
    app.launch(server_name=args.host, server_port=int(args.port), share=bool(args.share), show_api=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
