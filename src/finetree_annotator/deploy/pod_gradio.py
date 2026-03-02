from __future__ import annotations

import argparse
import html
import json
import os
import secrets
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, Field

_DEFAULT_TINY_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6p6nQAAAAASUVORK5CYII="
)


class PlaygroundChatRequest(BaseModel):
    system_prompt: str = ""
    user_prompt: str = ""
    model: str = "qwen-gt"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    max_tokens: int = Field(default=120, ge=1, le=4096)
    image_data_url: Optional[str] = None


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


def _image_file_to_data_url(image_path: Path) -> str:
    import base64
    import mimetypes

    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _resolve_basic_auth_credentials() -> tuple[str, str]:
    username = str(os.getenv("FINETREE_PLAYGROUND_USER") or os.getenv("FINETREE_GRADIO_USER") or "").strip()
    password = str(os.getenv("FINETREE_PLAYGROUND_PASS") or os.getenv("FINETREE_GRADIO_PASS") or "").strip()
    if not username or not password:
        raise RuntimeError(
            "Playground auth is required. "
            "Set FINETREE_PLAYGROUND_USER/FINETREE_PLAYGROUND_PASS "
            "or FINETREE_GRADIO_USER/FINETREE_GRADIO_PASS."
        )
    return username, password


def _resolve_pod_api_key() -> str:
    api_key = str(os.getenv("FINETREE_POD_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("FINETREE_POD_API_KEY is required for playground-to-API calls.")
    return api_key


def _resolve_api_base_url(explicit_api_base_url: Optional[str]) -> str:
    candidate = str(explicit_api_base_url or os.getenv("FINETREE_PLAYGROUND_API_BASE_URL") or "http://127.0.0.1:6666")
    return candidate.rstrip("/")


def _extract_assistant_text(payload: dict[str, Any]) -> str:
    try:
        choice = payload["choices"][0]
    except Exception:
        return json.dumps(payload, ensure_ascii=False)

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    return json.dumps(payload, ensure_ascii=False)


def _build_openai_chat_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    system_prompt = str(request_payload.get("system_prompt") or "").strip()
    user_prompt = str(request_payload.get("user_prompt") or "").strip() or "Describe this image."
    model = str(request_payload.get("model") or "").strip() or "qwen-gt"
    image_data_url = str(request_payload.get("image_data_url") or "").strip() or _DEFAULT_TINY_IMAGE_DATA_URL

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(request_payload.get("max_tokens") or 120),
    }

    temperature = float(request_payload.get("temperature") or 0.7)
    top_p = float(request_payload.get("top_p") or 0.8)
    if temperature > 0.0:
        payload["temperature"] = temperature
        payload["top_p"] = top_p

    return payload


def _post_json(url: str, body: dict[str, Any], api_key: str, timeout_sec: float) -> dict[str, Any]:
    req = urllib_request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstream API error {exc.code}: {raw[:1000]}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Failed to reach upstream API: {exc}") from exc


def _iter_sse_data_events(url: str, body: dict[str, Any], api_key: str, timeout_sec: float) -> Iterator[str]:
    req = urllib_request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as response:
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        yield "\n".join(data_lines)
                        data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                field, sep, value = line.partition(":")
                if not sep or field != "data":
                    continue
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
            if data_lines:
                yield "\n".join(data_lines)
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstream API error {exc.code}: {raw[:1000]}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Failed to reach upstream API: {exc}") from exc


def _playground_html(*, default_model: str, default_image_data_url: Optional[str]) -> str:
    default_image_js = json.dumps(default_image_data_url or _DEFAULT_TINY_IMAGE_DATA_URL)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FineTree Pod Playground</title>
  <style>
    :root {{ --bg:#f6f4ef; --panel:#fffefb; --ink:#1f2430; --accent:#0b6e4f; --line:#d8d3c4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background:linear-gradient(145deg,#f6f4ef,#ece7d8); color:var(--ink); }}
    .wrap {{ max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
    .card {{ background: var(--panel); border:1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 8px 28px rgba(21,29,41,0.08); }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}
    label {{ font-weight: 600; font-size: 13px; display:block; margin-bottom: 6px; }}
    input, textarea, button {{ width:100%; border:1px solid var(--line); border-radius: 10px; padding:10px; font: inherit; background:white; }}
    textarea {{ min-height: 90px; resize: vertical; }}
    .row {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }}
    .status {{ font-size: 12px; min-height: 20px; margin-top: 8px; color:#6b7280; }}
    .out {{ min-height: 240px; white-space: pre-wrap; }}
    .run {{ background: var(--accent); color:white; border-color: var(--accent); font-weight:700; cursor:pointer; }}
    .run:disabled {{ opacity:0.7; cursor:not-allowed; }}
    @media (max-width: 900px) {{ .grid, .row {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2>FineTree Pod Playground (5555)</h2>
      <div class="grid">
        <div>
          <label>System Prompt</label>
          <textarea id="systemPrompt">You are a precise OCR/extraction assistant.</textarea>
          <label>User Prompt</label>
          <textarea id="userPrompt">Describe the page briefly.</textarea>
          <label>Image (optional)</label>
          <input id="imageInput" type="file" accept="image/*" />
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Model</label>
              <input id="model" value="{html.escape(default_model)}" />
            </div>
            <div>
              <label>Temperature</label>
              <input id="temperature" type="number" step="0.01" min="0" max="2" value="0.7" />
            </div>
            <div>
              <label>Top P</label>
              <input id="topP" type="number" step="0.01" min="0" max="1" value="0.8" />
            </div>
          </div>
          <div style="margin-top:10px;">
            <label>Max Tokens</label>
            <input id="maxTokens" type="number" min="1" max="4096" value="120" />
          </div>
          <div style="margin-top:14px;">
            <button class="run" id="runBtn">Run</button>
            <div id="status" class="status"></div>
          </div>
        </div>
        <div>
          <label>Assistant Output</label>
          <textarea id="output" class="out" readonly></textarea>
        </div>
      </div>
    </div>
  </div>
  <script>
    let imageDataUrl = {default_image_js};
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const runBtn = document.getElementById("runBtn");
    document.getElementById("imageInput").addEventListener("change", async (e) => {{
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {{ imageDataUrl = String(reader.result || ""); statusEl.textContent = "Image loaded."; }};
      reader.readAsDataURL(file);
    }});
    runBtn.addEventListener("click", async () => {{
      runBtn.disabled = true;
      statusEl.textContent = "Streaming...";
      outputEl.value = "";
      const payload = {{
        system_prompt: document.getElementById("systemPrompt").value,
        user_prompt: document.getElementById("userPrompt").value,
        model: document.getElementById("model").value,
        temperature: Number(document.getElementById("temperature").value || 0.7),
        top_p: Number(document.getElementById("topP").value || 0.8),
        max_tokens: Number(document.getElementById("maxTokens").value || 120),
        image_data_url: imageDataUrl
      }};
      try {{
        const resp = await fetch("/api/chat/stream", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
        if (!resp.ok) {{
          const message = await resp.text();
          throw new Error(message || `HTTP ${resp.status}`);
        }}
        if (!resp.body) {{
          throw new Error("Response body is unavailable for streaming.");
        }}

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let doneSeen = false;
        let buffer = "";

        const parseEvent = (eventText) => {{
          const lines = eventText.split(/\\r?\\n/);
          const dataLines = [];
          for (const line of lines) {{
            if (line.startsWith("data:")) {{
              dataLines.push(line.slice(5).trimStart());
            }}
          }}
          if (dataLines.length === 0) return;
          const data = dataLines.join("\\n");
          if (data === "[DONE]") {{
            doneSeen = true;
            return;
          }}
          let payload;
          try {{
            payload = JSON.parse(data);
          }} catch (_err) {{
            return;
          }}
          if (payload.error) {{
            const message = payload.error.message || payload.error || "Unknown streaming error";
            throw new Error(String(message));
          }}
          const choice = (payload.choices && payload.choices[0]) || null;
          const token = choice && choice.delta && typeof choice.delta.content === "string"
            ? choice.delta.content
            : null;
          if (token) {{
            outputEl.value += token;
            outputEl.scrollTop = outputEl.scrollHeight;
          }}
        }};

        while (true) {{
          const {{ value, done }} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {{ stream: true }});
          let match = buffer.match(/\\r?\\n\\r?\\n/);
          while (match) {{
            const boundary = match.index ?? -1;
            if (boundary < 0) break;
            const eventText = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + match[0].length);
            parseEvent(eventText);
            match = buffer.match(/\\r?\\n\\r?\\n/);
          }}
        }}
        buffer += decoder.decode();
        if (buffer.trim()) {{
          parseEvent(buffer.trim());
        }}
        statusEl.textContent = doneSeen ? "Done." : "Stream closed.";
      }} catch (err) {{
        statusEl.textContent = "Error.";
        outputEl.value = String(err);
      }} finally {{
        runBtn.disabled = false;
      }}
    }});
  </script>
</body>
</html>"""


def create_app(*, api_base_url: Optional[str] = None, default_model: Optional[str] = None) -> Any:
    try:
        import secrets as py_secrets
        from fastapi import Depends, FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.security import HTTPBasic, HTTPBasicCredentials
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pod playground requires fastapi, pydantic and uvicorn.") from exc

    username, password = _resolve_basic_auth_credentials()
    pod_api_key = _resolve_pod_api_key()
    resolved_base_url = _resolve_api_base_url(api_base_url)
    timeout_sec = float(str(os.getenv("FINETREE_PLAYGROUND_TIMEOUT_SEC") or "600"))
    served_model = str(default_model or os.getenv("FINETREE_SERVED_MODEL_NAME") or "qwen-gt").strip()

    example_path = _default_example_image_path()
    default_image_data_url = _image_file_to_data_url(example_path) if example_path else None

    security = HTTPBasic(auto_error=False)
    app = FastAPI(title="FineTree Pod Playground", docs_url=None, redoc_url=None, openapi_url=None)

    def _require_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:
        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )
        valid_user = py_secrets.compare_digest(credentials.username, username)
        valid_pass = py_secrets.compare_digest(credentials.password, password)
        if not (valid_user and valid_pass):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

    @app.get("/", response_class=HTMLResponse)
    async def index(_auth: None = Depends(_require_auth)) -> str:
        return _playground_html(default_model=served_model, default_image_data_url=default_image_data_url)

    @app.post("/api/chat")
    async def chat(payload: PlaygroundChatRequest, _auth: None = Depends(_require_auth)) -> Any:
        request_payload = _build_openai_chat_payload(payload.model_dump())
        url = f"{resolved_base_url}/v1/chat/completions"
        try:
            upstream = _post_json(url=url, body=request_payload, api_key=pod_api_key, timeout_sec=timeout_sec)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
        return JSONResponse(
            {
                "ok": True,
                "assistant_text": _extract_assistant_text(upstream),
                "response": upstream,
            }
        )

    @app.post("/api/chat/stream")
    async def chat_stream(payload: PlaygroundChatRequest, _auth: None = Depends(_require_auth)) -> Any:
        request_payload = _build_openai_chat_payload(payload.model_dump())
        request_payload["stream"] = True
        url = f"{resolved_base_url}/v1/chat/completions"

        def _proxy_stream() -> Iterator[bytes]:
            done_sent = False
            try:
                for event_data in _iter_sse_data_events(
                    url=url,
                    body=request_payload,
                    api_key=pod_api_key,
                    timeout_sec=timeout_sec,
                ):
                    if not event_data:
                        continue
                    if event_data == "[DONE]":
                        done_sent = True
                        yield b"data: [DONE]\n\n"
                        break
                    yield f"data: {event_data}\n\n".encode("utf-8")
            except Exception as exc:
                err = {"error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")
            finally:
                if not done_sent:
                    yield b"data: [DONE]\n\n"

        return StreamingResponse(_proxy_stream(), media_type="text/event-stream")

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineTree Pod Playground app.")
    parser.add_argument("--config", default=None, help="FineTree YAML config path (kept for compatibility).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5555, help="Bind port (default: 5555).")
    parser.add_argument("--api-base-url", default=None, help="Internal pod API base URL (default: http://127.0.0.1:6666).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pod playground requires uvicorn. Install with `pip install uvicorn`.") from exc

    app = create_app(api_base_url=args.api_base_url)
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
