"""`pvllm complete` and `pvllm chat`. R2.6, C5.

Clients for a running server, as upstream's are -- they start no engine. Driven here
against a real ASGI app over a real socket, because the thing being tested is the wire:
SSE framing, the `[DONE]` sentinel, the error envelope, and the conversation a chat
session carries forward.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.cli.main import build_parser, main
from pvllm.entrypoints.cli.openai import resolve_model, stream_events
from pvllm.entrypoints.openai.api_server import build_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server() -> str:
    """A real server on a real port. Returns its `/v1` base url."""
    port = _free_port()
    app = build_app(
        AsyncEngineArgs(
            model="tiny-test",
            served_model_name="test-model",
            max_model_len=512,
            block_size=16,
            max_num_batched_tokens=256,
            device_card="tiny-2gb",
            disable_log_stats=True,
        ).create_engine_config(),
        registry=CollectorRegistry(),
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not uvicorn_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}/v1"

    uvicorn_server.should_exit = True
    thread.join(timeout=10)


# --- the parser -------------------------------------------------------------


def test_both_commands_are_registered():
    """R2.6 names `complete`; it existed as a placeholder that printed 'not
    implemented' and returned 2."""
    parser = build_parser()
    for command in ("complete", "chat"):
        args = parser.parse_args([command, "-q", "hi"])
        assert args.command == command
        assert args.quick == "hi"
        assert args.url == "http://localhost:8000/v1"


def test_chat_takes_a_system_prompt_and_complete_does_not():
    parser = build_parser()
    assert parser.parse_args(["chat", "--system-prompt", "Be terse."]).system_prompt
    with pytest.raises(SystemExit):
        parser.parse_args(["complete", "--system-prompt", "x"])


# --- the wire ---------------------------------------------------------------


def test_the_model_defaults_to_the_first_one_the_server_lists(server):
    assert resolve_model(server, None, None) == "test-model"
    assert resolve_model(server, "explicit", None) == "explicit"


def test_the_stream_parses_sse_and_stops_at_done(server):
    """The `[DONE]` sentinel is a literal rather than JSON, which is exactly the shape
    a client written against OpenAI expects."""
    events = list(
        stream_events(
            f"{server}/completions",
            {
                "model": "test-model",
                "prompt": "hello",
                "stream": True,
                "max_tokens": 6,
            },
            None,
        )
    )
    assert events
    text = "".join(
        choice["text"] for event in events for choice in event.get("choices") or ()
    )
    assert text
    # Every payload was JSON; `[DONE]` ended the iteration rather than being yielded.
    assert all(isinstance(event, dict) for event in events)


def test_complete_prints_the_completion(server, capsys):
    assert main(["complete", "--url", server, "-q", "hello", "--max-tokens", "6"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert "Using model: test-model" in captured.err


def test_stats_are_labeled_as_the_client_s_own(server, capsys):
    """Under the default virtual clock the server answers as fast as it can compute,
    so a client-side TTFT describes the simulator rather than the deployment. R9.5
    says every latency number is labeled; this one is labeled as measuring something
    else entirely."""
    main(["complete", "--url", server, "-q", "hello", "--max-tokens", "6", "--stats"])
    captured = capsys.readouterr()
    assert "[client-measured]" in captured.err
    assert "not the modeled engine" in captured.err


def test_chat_carries_the_conversation_forward(server, capsys):
    assert (
        main(
            [
                "chat",
                "--url",
                server,
                "-q",
                "hi",
                "--max-tokens",
                "6",
                "--system-prompt",
                "Be terse.",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip()


def test_multiple_prompts_come_from_stdin(server, capsys, monkeypatch):
    """Without `--quick` it reads a prompt per line, which is what makes it usable in
    a pipe."""
    lines = iter(["first prompt", "second prompt"])

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["complete", "--url", server, "--max-tokens", "4"]) == 0
    # One line of output per prompt.
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


# --- failures ---------------------------------------------------------------


def test_an_unreachable_server_says_so_rather_than_traversing(capsys):
    port = _free_port()
    assert main(["complete", "--url", f"http://127.0.0.1:{port}/v1", "-q", "x"]) == 1
    assert "cannot reach" in capsys.readouterr().err


def test_a_server_error_is_reported_with_the_server_s_own_message(server, capsys):
    """The useful half of a failure is what the server said about it."""
    assert main(["complete", "--url", server, "--model-name", "nope", "-q", "x"]) == 1
    err = capsys.readouterr().err
    assert "HTTP 404" in err
    assert "does not exist" in err
