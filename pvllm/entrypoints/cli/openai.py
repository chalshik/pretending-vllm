"""`pvllm complete` and `pvllm chat`: clients for a running server. R2.6.

Upstream: vllm/entrypoints/cli/openai.py
Tier: B

Both talk to a server over the OpenAI API rather than starting an engine, which is
what upstream's do and what a user's muscle memory expects: `pvllm serve` in one
terminal, `pvllm complete` in another.

Upstream reaches for the `openai` package. This uses the standard library instead, so
the commands work on a bare install -- a CLI that cannot run without an extra is a CLI
that is not there when you want it. The wire format is the contract either way (C5),
and the request bodies here are the ones the `openai` client sends.

**`--stats` measures the client, not the engine.** TTFT and tokens-per-second come
from a wall clock on this side of the socket, and under the default virtual clock the
server answers as fast as it can compute, so those numbers describe the simulator's
speed rather than the deployment's. The engine's own modeled numbers are on `/metrics`
and in the trace (R9.5). Run the server with `--clock-mode real` if you want the two to
agree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

#: Long enough that a queued request behind a long prefill is not mistaken for a dead
#: server, short enough that a wrong `--url` fails while the user is still watching.
_TIMEOUT_SECONDS = 300.0


class StreamFailure(RuntimeError):
    """The server reported an error *inside* an already-200 stream."""


def add_query_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The options both commands share, with upstream's names and defaults."""
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1",
        help="url of the running OpenAI-compatible server",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="model to request; defaults to the first one /v1/models lists",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="sent as a bearer token. The server does not check it; it is here so a "
        "proxy in front of it can.",
    )
    parser.add_argument(
        "-q",
        "--quick",
        metavar="PROMPT",
        default=None,
        help="send one prompt, print the response, and exit",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print client-side TTFT and tokens/sec. These time this process, not "
        "the engine -- see /metrics for the modeled numbers.",
    )
    return parser


def _request(
    url: str, payload: dict[str, Any] | None, api_key: str | None, stream: bool
) -> Any:
    """POST (or GET) and return the raw response object."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)


def _server_error(exc: urllib.error.HTTPError) -> str:
    """The server's own message, which is the useful half of a failure."""
    try:
        body = json.loads(exc.read().decode())
    except (ValueError, OSError):
        return f"HTTP {exc.code}"
    error = body.get("error")
    if isinstance(error, dict) and error.get("message"):
        return f"HTTP {exc.code}: {error['message']}"
    return f"HTTP {exc.code}"


def resolve_model(url: str, model_name: str | None, api_key: str | None) -> str:
    """The model to request. Asks the server when the user did not say."""
    if model_name:
        return model_name
    with _request(f"{url.rstrip('/')}/models", None, api_key, stream=False) as response:
        listed = json.loads(response.read().decode())
    entries = listed.get("data") or []
    if not entries:
        raise RuntimeError(f"{url} lists no models")
    return str(entries[0]["id"])


def stream_events(
    url: str, payload: dict[str, Any], api_key: str | None
) -> Iterator[dict[str, Any]]:
    """Yield SSE payloads until `[DONE]`.

    Server-sent events are `data: <json>` lines separated by blanks, and the terminal
    `[DONE]` is a literal rather than JSON -- which is exactly the shape a client
    written against OpenAI expects, and the reason it is reproduced rather than
    simplified.
    """
    with _request(url, payload, api_key, stream=True) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                return
            event = json.loads(body)
            # A stream that fails after its headers cannot use a status code, so the
            # server sends the error as an SSE payload and then `[DONE]`. Dropping it
            # printed a truncated answer and exited 0 -- a script could not tell a
            # failed generation from a finished one, and `chat` kept the half-written
            # reply as context for the next turn.
            error = event.get("error")
            if isinstance(error, dict):
                raise StreamFailure(str(error.get("message") or "stream failed"))
            yield event


def _print_stream(
    events: Iterator[dict[str, Any]], chat: bool, stats: bool
) -> tuple[str, float | None, int]:
    """Print deltas as they arrive. Returns `(text, ttft, completion_tokens)`."""
    output: list[str] = []
    started = time.perf_counter()
    ttft: float | None = None
    completion_tokens = 0

    for event in events:
        usage = event.get("usage")
        if usage:
            completion_tokens = int(usage.get("completion_tokens") or 0)
        for choice in event.get("choices") or ():
            piece = (
                (choice.get("delta") or {}).get("content")
                if chat
                else choice.get("text")
            )
            if not piece:
                continue
            if ttft is None:
                ttft = time.perf_counter() - started
            output.append(piece)
            print(piece, end="", flush=True)
    print()

    if stats:
        elapsed = time.perf_counter() - started
        tokens = completion_tokens or len(output)
        rate = tokens / elapsed if elapsed > 0 else 0.0
        # Labeled every time, because the number is honest about a different thing
        # than the reader expects: it is this process's wall clock, and under the
        # default virtual clock the engine spends no real time at all.
        # `ttft` is None when the response carried no text at all -- a request
        # stopped on its first token, say. Reported as absent rather than as zero,
        # which would read as an impossibly fast first token.
        first_token = f"{ttft * 1000:.1f}ms" if ttft is not None else "n/a"
        print(
            f"[client-measured] ttft={first_token} "
            f"tokens={tokens} rate={rate:.1f} tok/s "
            f"-- times this client, not the modeled engine; see /metrics",
            file=sys.stderr,
        )
    return "".join(output), ttft, completion_tokens


def _prompts(quick: str | None) -> Iterator[str]:
    """One prompt from `--quick`, or a line at a time from stdin."""
    if quick is not None:
        yield quick
        return
    if sys.stdin.isatty():
        print("Enter a prompt (Ctrl-D to exit):", file=sys.stderr)
    while True:
        try:
            line = input("> " if sys.stdin.isatty() else "")
        except EOFError:
            return
        if line:
            yield line


def run_complete(args: argparse.Namespace) -> int:
    """`pvllm complete`: text completions against a running server."""
    url = args.url.rstrip("/")
    try:
        model = resolve_model(url, args.model_name, args.api_key)
    except urllib.error.HTTPError as exc:
        print(_server_error(exc), file=sys.stderr)
        return 1
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        print(f"cannot reach {url}: {exc}", file=sys.stderr)
        return 1

    print(f"Using model: {model}", file=sys.stderr)
    for prompt in _prompts(args.quick):
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.stats:
            payload["stream_options"] = {"include_usage": True}
        try:
            _print_stream(
                stream_events(f"{url}/completions", payload, args.api_key),
                chat=False,
                stats=args.stats,
            )
        except urllib.error.HTTPError as exc:
            print(_server_error(exc), file=sys.stderr)
            return 1
        except StreamFailure as exc:
            print(f"stream failed: {exc}", file=sys.stderr)
            return 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # The same handling the model lookup gets. Only the *first* request was
            # guarded before, so a server restarted between two piped prompts came
            # out as a raw urllib traceback -- the thing this module says it turns
            # into a message.
            print(f"cannot reach {url}: {exc}", file=sys.stderr)
            return 1
    return 0


def run_chat(args: argparse.Namespace) -> int:
    """`pvllm chat`: chat completions, carrying the conversation forward."""
    url = args.url.rstrip("/")
    try:
        model = resolve_model(url, args.model_name, args.api_key)
    except urllib.error.HTTPError as exc:
        print(_server_error(exc), file=sys.stderr)
        return 1
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        print(f"cannot reach {url}: {exc}", file=sys.stderr)
        return 1

    print(f"Using model: {model}", file=sys.stderr)
    conversation: list[dict[str, str]] = []
    if args.system_prompt:
        conversation.append({"role": "system", "content": args.system_prompt})

    for message in _prompts(args.quick):
        conversation.append({"role": "user", "content": message})
        payload: dict[str, Any] = {
            "model": model,
            "messages": conversation,
            "stream": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.stats:
            payload["stream_options"] = {"include_usage": True}
        try:
            reply, _, _ = _print_stream(
                stream_events(f"{url}/chat/completions", payload, args.api_key),
                chat=True,
                stats=args.stats,
            )
        except urllib.error.HTTPError as exc:
            print(_server_error(exc), file=sys.stderr)
            return 1
        except StreamFailure as exc:
            # The half-written reply is *not* appended: carrying a truncated turn
            # forward would poison every later turn's context with text the model
            # never finished saying.
            conversation.pop()
            print(f"stream failed: {exc}", file=sys.stderr)
            return 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            conversation.pop()
            print(f"cannot reach {url}: {exc}", file=sys.stderr)
            return 1
        # Kept, so the next turn carries the context -- which is also what makes the
        # prefix cache hit, and therefore what makes a multi-turn session here behave
        # like one in production.
        conversation.append({"role": "assistant", "content": reply})
    return 0


__all__ = [
    "add_query_args",
    "resolve_model",
    "run_chat",
    "run_complete",
    "stream_events",
]
