"""Server-Sent Events helpers for streaming responses."""
from __future__ import annotations

import json
from typing import Any

from fastapi.responses import StreamingResponse


def sse_event(data: Any, *, event: str | None = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    parts: list[str] = []
    if event:
        parts.append(f"event: {event}")
    parts.append(f"data: {payload}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)


def sse_response(generator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
