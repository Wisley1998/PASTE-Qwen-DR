#!/usr/bin/env python3
"""Send one real OpenAI-compatible chat completion to the vLLM server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL_ID = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"


def parse_args() -> argparse.Namespace:
    host = os.environ.get("VLLM_PROBE_HOST", os.environ.get("VLLM_HOST", "127.0.0.1"))
    port = os.environ.get("VLLM_PORT", "8000")
    parser = argparse.ArgumentParser(
        description="Run and validate one real /v1/chat/completions request."
    )
    parser.add_argument("--base-url", default=f"http://{host}:{port}")
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"smoke test failed: {message}", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        return fail("--timeout must be positive")
    if args.max_tokens <= 0:
        return fail("--max-tokens must be positive")

    request_body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with a short confirmation that this inference server works.",
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            status = response.status
            raw_body = response.read()
    except HTTPError as exc:
        return fail(f"HTTP status {exc.code} from {endpoint}")
    except URLError as exc:
        return fail(f"could not reach {endpoint}: {exc.reason}")
    except TimeoutError:
        return fail(f"request timed out after {args.timeout:g}s")

    if status != 200:
        return fail(f"expected HTTP 200, got {status}")
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return fail(f"response was not valid JSON: {exc}")
    if not isinstance(payload, dict):
        return fail("response JSON is not an object")
    if payload.get("model") != args.model:
        return fail(f"response model {payload.get('model')!r} != {args.model!r}")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return fail("response has no choices[0].message.content")
    if not isinstance(content, str) or not content.strip():
        return fail("assistant content is empty")

    result = {
        "ok": True,
        "http_status": status,
        "model": payload["model"],
        "content_characters": len(content),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
