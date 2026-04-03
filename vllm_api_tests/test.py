#!/usr/bin/env python3

import argparse
import json

def _normalize_base_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url[: -len("/chat/completions")]
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _resolve_model(client, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model

    models = list(client.models.list().data)
    if not models:
        raise SystemExit("No models returned by /v1/models. Pass --model explicitly.")

    model_id = str(getattr(models[0], "id", "") or "").strip()
    if not model_id:
        raise SystemExit("First /v1/models entry has no id. Pass --model explicitly.")
    return model_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal OpenAI-compatible chat completion smoke test."
    )
    parser.add_argument("base_url", help="OpenAI-compatible base URL or chat completions URL")
    parser.add_argument("--model", help="Model id. If omitted, the script uses /v1/models")
    parser.add_argument("--prompt", default="Reply with exactly: ok", help="User prompt")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max completion tokens")
    args = parser.parse_args()

    from openai import OpenAI

    base_url = _normalize_base_url(args.base_url)
    client = OpenAI(
        base_url=base_url,
        api_key="unused",
        timeout=120.0,
    )
    model = _resolve_model(client, args.model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=0,
        stream=False,
    )

    print(f"base_url={base_url}")
    print(f"model={model}")
    choice = response.choices[0].message if response.choices else None
    content = choice.content if choice else None
    if isinstance(content, str):
        print("assistant:")
        print(content)
        return 0

    print("raw_response:")
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
