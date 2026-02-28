from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..qwen_vlm import generate_content_from_image, generate_page_extraction_from_image

DEFAULT_PROMPT = "Extract page JSON using the FineTree schema."
_SUPPORTED_RESPONSE_MODES = {"page_extraction", "text"}


def _normalize_response_mode(job_input: Dict[str, Any]) -> str:
    response_mode = str(job_input.get("response_mode") or "").strip().lower()

    # Backward-compatible flag: parse_json=false -> text output.
    if not response_mode and "parse_json" in job_input:
        parse_json = bool(job_input.get("parse_json"))
        response_mode = "page_extraction" if parse_json else "text"

    if not response_mode:
        response_mode = "page_extraction"

    if response_mode not in _SUPPORTED_RESPONSE_MODES:
        raise ValueError(
            f"Unsupported response_mode={response_mode!r}. "
            f"Supported values: {sorted(_SUPPORTED_RESPONSE_MODES)}."
        )
    return response_mode


def _decode_base64_payload(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def _suffix_from_mime(mime_type: str) -> str:
    suffix = mimetypes.guess_extension(mime_type) or ".png"
    if suffix == ".jpe":
        return ".jpg"
    return suffix


def _write_temp_image(image_bytes: bytes, mime_type: str, temp_dir: Path) -> Path:
    image_path = temp_dir / f"input{_suffix_from_mime(mime_type)}"
    image_path.write_bytes(image_bytes)
    return image_path


def _image_path_from_data_uri(raw_data_uri: str, temp_dir: Path) -> Path:
    if not raw_data_uri.startswith("data:") or ";base64," not in raw_data_uri:
        raise ValueError("image_data_uri must be a base64 RFC2397 data URI.")
    header, payload = raw_data_uri.split(",", 1)
    mime_type = header[5:].split(";", 1)[0].strip() or "image/png"
    image_bytes = _decode_base64_payload(payload)
    return _write_temp_image(image_bytes=image_bytes, mime_type=mime_type, temp_dir=temp_dir)


def _resolve_image_path(job_input: Dict[str, Any], temp_dir: Path) -> Path:
    raw_image_path = str(job_input.get("image_path") or "").strip()
    if raw_image_path:
        image_path = Path(raw_image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"image_path not found: {image_path}")
        return image_path.resolve()

    raw_data_uri = str(job_input.get("image_data_uri") or "").strip()
    if raw_data_uri:
        return _image_path_from_data_uri(raw_data_uri=raw_data_uri, temp_dir=temp_dir)

    raw_base64 = str(job_input.get("image_base64") or "").strip()
    if raw_base64:
        mime_type = str(job_input.get("image_mime_type") or "image/png").strip() or "image/png"
        image_bytes = _decode_base64_payload(raw_base64)
        return _write_temp_image(image_bytes=image_bytes, mime_type=mime_type, temp_dir=temp_dir)

    raise ValueError("Input must include one of: image_path, image_data_uri, image_base64.")


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    return value


def run_inference(job_input: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(job_input, dict):
        raise TypeError("RunPod input must be a JSON object.")

    prompt = str(job_input.get("prompt") or DEFAULT_PROMPT)
    config_path = str(job_input.get("config_path") or "").strip() or None
    model = str(job_input.get("model") or "").strip() or None
    response_mode = _normalize_response_mode(job_input)

    with tempfile.TemporaryDirectory(prefix="finetree-runpod-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        image_path = _resolve_image_path(job_input=job_input, temp_dir=temp_dir)

        if response_mode == "text":
            text = generate_content_from_image(
                image_path=image_path,
                prompt=prompt,
                model=model,
                config_path=config_path,
            )
            return {"ok": True, "mode": "text", "text": text}

        extraction = generate_page_extraction_from_image(
            image_path=image_path,
            prompt=prompt,
            model=model,
            config_path=config_path,
        )
        return {"ok": True, "mode": "page_extraction", "result": _to_jsonable(extraction)}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        raise TypeError("RunPod event must be a JSON object.")
    if "input" in event:
        payload = event.get("input")
    else:
        payload = event
    if not isinstance(payload, dict):
        raise TypeError("RunPod event.input must be a JSON object.")
    return run_inference(payload)


def _load_test_payload(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Test input file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Test input JSON must be an object.")
    if "input" in raw:
        payload = raw["input"]
        if not isinstance(payload, dict):
            raise ValueError("Top-level input must be an object.")
        return payload
    return raw


def serve_forever() -> int:
    try:
        import runpod
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("runpod package is required. Install with `pip install runpod`.") from exc

    runpod.serverless.start({"handler": handler})
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineTree RunPod Serverless queue worker.")
    parser.add_argument("--serve", action="store_true", help="Run as RunPod Serverless worker.")
    parser.add_argument(
        "--test-input",
        default="deploy/runpod/test_input.json",
        help="Local test JSON payload path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional override for config_path during local test.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output in local test mode.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.serve:
        return serve_forever()

    payload = _load_test_payload(Path(args.test_input).expanduser())
    if args.config and "config_path" not in payload:
        payload["config_path"] = args.config
    result = run_inference(payload)
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
