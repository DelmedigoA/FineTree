from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import mimetypes
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ..qwen_vlm import generate_content_from_image, stream_content_from_image

DEFAULT_PROMPT = "Extract page JSON using the FineTree schema."


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
        raise ValueError("image_url.url must be a base64 RFC2397 data URI.")
    header, payload = raw_data_uri.split(",", 1)
    mime_type = header[5:].split(";", 1)[0].strip() or "image/png"
    image_bytes = _decode_base64_payload(payload)
    return _write_temp_image(image_bytes=image_bytes, mime_type=mime_type, temp_dir=temp_dir)


def _extract_prompt_and_image_url(payload: Dict[str, Any]) -> tuple[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")

    system_parts: list[str] = []
    user_parts: list[str] = []
    image_url: Optional[str] = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                if role == "system":
                    system_parts.append(content.strip())
                elif role == "user":
                    user_parts.append(content.strip())
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    if role == "system":
                        system_parts.append(text.strip())
                    elif role == "user":
                        user_parts.append(text.strip())
                continue
            if role == "user" and item_type == "image_url":
                raw_image_url = item.get("image_url")
                if isinstance(raw_image_url, dict):
                    url_value = raw_image_url.get("url")
                    if isinstance(url_value, str) and url_value.strip():
                        image_url = url_value.strip()

    user_prompt = "\n".join(part for part in user_parts if part).strip() or DEFAULT_PROMPT
    if system_parts:
        system_prompt = "\n".join(part for part in system_parts if part).strip()
        prompt = f"System instructions:\n{system_prompt}\n\nUser request:\n{user_prompt}"
    else:
        prompt = user_prompt

    if not image_url:
        raise ValueError("messages must include user content with an image_url item.")
    return prompt, image_url


def _extract_sampling_controls(payload: Dict[str, Any]) -> tuple[Optional[bool], Optional[float], Optional[float]]:
    do_sample_raw = payload.get("do_sample")
    temperature_raw = payload.get("temperature")
    top_p_raw = payload.get("top_p")

    do_sample: Optional[bool] = None
    if do_sample_raw is not None:
        if isinstance(do_sample_raw, bool):
            do_sample = do_sample_raw
        else:
            raise ValueError("do_sample must be a boolean when provided.")

    temperature: Optional[float] = None
    if temperature_raw is not None:
        try:
            temperature = float(temperature_raw)
        except Exception as exc:
            raise ValueError("temperature must be numeric when provided.") from exc
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0.")

    top_p: Optional[float] = None
    if top_p_raw is not None:
        try:
            top_p = float(top_p_raw)
        except Exception as exc:
            raise ValueError("top_p must be numeric when provided.") from exc
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the range (0, 1].")

    if do_sample is None and temperature is not None and temperature > 0.0:
        do_sample = True

    return do_sample, temperature, top_p


def _build_completion_chunk(
    request_id: str,
    model_name: str,
    *,
    content: str = "",
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    if content:
        delta["content"] = content
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _build_completion_response(request_id: str, model_name: str, content: str) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _bearer_token(auth_header: str) -> str:
    if not isinstance(auth_header, str):
        return ""
    prefix = "bearer "
    if auth_header.lower().startswith(prefix):
        return auth_header[len(prefix) :].strip()
    return ""


def _resolve_model_selection(payload_model: Any, served_model_name: str) -> tuple[str, Optional[str]]:
    requested = str(payload_model or "").strip()
    served = str(served_model_name or "").strip() or "qwen-gt"
    if not requested:
        return served, None
    if requested == served:
        # Clients typically send the served alias ("qwen-gt"). For local backend,
        # passing this as a model override would fail HF model resolution.
        return served, None
    return requested, requested


def create_app(
    *,
    config_path: Optional[str] = None,
    api_key: Optional[str] = None,
    max_concurrency: int = 2,
    served_model_name: Optional[str] = None,
) -> Any:
    try:
        import asyncio
        from fastapi import Body, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pod API requires fastapi and uvicorn. Install with `pip install fastapi uvicorn`.") from exc

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1.")

    required_api_key = str(api_key or os.getenv("FINETREE_POD_API_KEY") or "").strip()
    if not required_api_key:
        raise RuntimeError("FINETREE_POD_API_KEY is required for pod API auth.")

    cfg_path = str(config_path or os.getenv("FINETREE_QWEN_CONFIG") or "").strip() or None
    model_id = str(served_model_name or os.getenv("FINETREE_SERVED_MODEL_NAME") or "qwen-gt").strip()
    semaphore = asyncio.Semaphore(max_concurrency)

    app = FastAPI(title="FineTree Pod API", version="1.0.0")
    logger = logging.getLogger("finetree.pod_api")
    debug_errors = str(os.getenv("FINETREE_POD_DEBUG_ERRORS") or "").strip().lower() in {"1", "true", "yes", "on"}

    async def _check_auth(authorization: Optional[str]) -> None:
        provided = _bearer_token(str(authorization or ""))
        if not provided or not secrets.compare_digest(provided, required_api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"ok": True, "service": "finetree-pod-api"}

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        await _check_auth(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "finetree",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: Dict[str, Any] = Body(...),
        authorization: Optional[str] = Header(default=None),
    ) -> Any:
        await _check_auth(authorization)

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.001)
        except Exception:
            raise HTTPException(status_code=429, detail="Pod is busy. Retry shortly.")
        lease: dict[str, bool] = {"active": True}

        def _release_lease() -> None:
            if lease["active"]:
                semaphore.release()
                lease["active"] = False

        req_id = f"chatcmpl-{secrets.token_hex(8)}"
        stream = bool(payload.get("stream"))
        model_name, inference_model_override = _resolve_model_selection(payload.get("model"), model_id)
        max_tokens_int: Optional[int] = None
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            try:
                max_tokens_int = int(max_tokens)
            except Exception:
                _release_lease()
                raise HTTPException(status_code=400, detail="max_tokens must be an integer.")
            if max_tokens_int <= 0:
                _release_lease()
                raise HTTPException(status_code=400, detail="max_tokens must be > 0.")

        try:
            prompt, image_url = _extract_prompt_and_image_url(payload)
        except ValueError as exc:
            _release_lease()
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            do_sample_override, temperature_override, top_p_override = _extract_sampling_controls(payload)
        except ValueError as exc:
            _release_lease()
            raise HTTPException(status_code=400, detail=str(exc))

        def _stream_sse() -> Iterator[bytes]:
            try:
                with tempfile.TemporaryDirectory(prefix="finetree-pod-api-") as temp_dir_str:
                    temp_dir = Path(temp_dir_str)
                    image_path = _image_path_from_data_uri(image_url, temp_dir=temp_dir)
                    for token in stream_content_from_image(
                        image_path=image_path,
                        prompt=prompt,
                        model=inference_model_override,
                        config_path=cfg_path,
                        max_new_tokens=max_tokens_int,
                        do_sample=do_sample_override,
                        temperature=temperature_override,
                        top_p=top_p_override,
                    ):
                        if not token:
                            continue
                        chunk = _build_completion_chunk(req_id, model_name, content=token)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                    final_chunk = _build_completion_chunk(req_id, model_name, finish_reason="stop")
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
            except ValueError as exc:
                err = {"error": {"message": str(exc), "type": "invalid_request_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                err_id = f"poderr-{secrets.token_hex(6)}"
                logger.exception(
                    "chat_completions stream failure id=%s model=%s override=%s stream=%s",
                    err_id,
                    model_name,
                    inference_model_override,
                    stream,
                )
                message = f"Internal server error. error_id={err_id}"
                if debug_errors:
                    message = f"{message} detail={repr(exc)}"
                err = {"error": {"message": message, "type": "server_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            finally:
                _release_lease()

        if stream:
            return StreamingResponse(_stream_sse(), media_type="text/event-stream")

        def _run_non_stream_inference() -> str:
            # Run blocking model load/inference off the event loop so health checks stay responsive.
            with tempfile.TemporaryDirectory(prefix="finetree-pod-api-") as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                image_path = _image_path_from_data_uri(image_url, temp_dir=temp_dir)
                return generate_content_from_image(
                    image_path=image_path,
                    prompt=prompt,
                    model=inference_model_override,
                    config_path=cfg_path,
                    max_new_tokens=max_tokens_int,
                    do_sample=do_sample_override,
                    temperature=temperature_override,
                    top_p=top_p_override,
                )

        try:
            text = await asyncio.to_thread(_run_non_stream_inference)
        except ValueError as exc:
            _release_lease()
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            _release_lease()
            err_id = f"poderr-{secrets.token_hex(6)}"
            logger.exception(
                "chat_completions failure id=%s model=%s override=%s stream=%s",
                err_id,
                model_name,
                inference_model_override,
                stream,
            )
            detail = f"Internal server error. error_id={err_id}"
            if debug_errors:
                detail = f"{detail} detail={repr(exc)}"
            raise HTTPException(status_code=500, detail=detail)
        finally:
            _release_lease()

        return JSONResponse(_build_completion_response(req_id, model_name, text))

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineTree Pod API service (OpenAI-compatible).")
    parser.add_argument("--config", default=None, help="FineTree YAML config path.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=6666, help="Bind port (default: 6666).")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("FINETREE_MAX_CONCURRENCY", "2")),
        help="Max concurrent inference requests.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name exposed by /v1/models and completion payloads.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pod API requires uvicorn. Install with `pip install uvicorn`.") from exc

    app = create_app(
        config_path=args.config,
        max_concurrency=int(args.max_concurrency),
        served_model_name=args.served_model_name,
    )
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
