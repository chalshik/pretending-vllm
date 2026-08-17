"""Exception handlers, so every error leaves in vLLM's envelope. C5, C7.

Upstream: vllm/entrypoints/serve/utils/server_utils.py
Tier: B

Without these, FastAPI answers a malformed body itself: HTTP 422 with its own
`{"detail": [...]}` shape. Real vLLM installs handlers that convert the same failure
into **400** with `{"error": {message, type, param, code}}` -- so a client that sends a
bad request is the one case where an unmodified product could tell pvllm from the real
thing, on *every* endpoint at once. That is what this module closes.

Two different `type` strings are correct here, and they are easy to conflate:

* a pydantic *schema* failure (a missing field, a wrong type) becomes
  `HTTPStatus.BAD_REQUEST.phrase`, which is the string ``"Bad Request"``;
* an error raised inside handler code goes through `create_error_response`, whose
  default is ``"BadRequestError"``.

Both appear on the wire, on different paths.
"""

from __future__ import annotations

import re
from http import HTTPStatus

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from pvllm.entrypoints.serve.utils.api_utils import sanitize_message
from pvllm.entrypoints.serve.utils.error_response import (
    ErrorInfo,
    ErrorResponse,
    to_error_response,
)

#: Any bracketed segment is a pydantic-core construct rather than a field name.
_BRACKETED_INTERNAL_RE = re.compile(r"[\[\]{}()]")

#: pydantic-core's internal schema-kind vocabulary. Not a stable public API: it can
#: grow when pydantic-core adds wrapper kinds, and the way to refresh it is to send
#: deliberately malformed values for union-typed fields (`stop`, `prompt`, `input`)
#: and look at the raw `loc` tuples that come back.
_INTERNAL_LOC_MARKERS = frozenset(
    {
        "function-wrap",
        "function-after",
        "function-before",
        "function-plain",
        "json-or-python",
        "lax-or-strict",
        "chain",
        "default",
        "nullable",
        "tagged-union",
        "union",
        "call",
        "arguments",
        "is-instance",
        "is-subclass",
        "callable",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "complex",
        "none",
        "nonetype",
    }
)


def _is_internal_loc_segment(segment: str) -> bool:
    if _BRACKETED_INTERNAL_RE.search(segment):
        return True
    return segment.lower() in _INTERNAL_LOC_MARKERS


def clean_loc_for_param(loc: tuple[object, ...]) -> str:
    """Join a pydantic error `loc` into the dotted path a client would recognise.

    `('body', 'function-wrap[__log_extra_fields__()]', 'prompt')` is `"body.prompt"`,
    not the raw join -- the middle segment names a pydantic wrapper, not a field
    anyone sent.
    """
    parts = [str(part) for part in loc if not _is_internal_loc_segment(str(part))]
    if not parts:
        return ".".join(str(part) for part in loc)
    return ".".join(parts)


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """A malformed body is a 400 in vLLM's envelope, not FastAPI's 422.

    **The message must be rebuilt from `exc.errors()`, never from `str(exc)`.** On the
    pinned stack (fastapi 0.141.1, pydantic 2.13.4) `str(exc)` appends FastAPI's
    endpoint context, which for a route defined as a closure inside `build_app` -- as
    every route here is -- includes the absolute source path, the line number and the
    function name:

        File "/…/pvllm/entrypoints/openai/api_server.py", line 202, in create_completion

    An earlier version of this docstring claimed that leak "does not reproduce on the
    pydantic pinned here". It was checked against a toy app whose endpoint was a
    module-level function, where it does not; against this app it does. Simplifying
    the construction below back to `str(exc)` reintroduces an absolute-path disclosure
    on every malformed body, on every endpoint.
    """
    assert isinstance(exc, RequestValidationError)
    param: str | None = None
    errors = exc.errors()
    if errors:
        loc = errors[0].get("loc") if isinstance(errors[0], dict) else None
        if loc:
            param = clean_loc_for_param(tuple(loc))

    if errors:
        label = "error" if len(errors) == 1 else "errors"
        message = f"{len(errors)} validation {label}:\n"
        message += "".join(f"  {error}\n" for error in errors)
        message = message.rstrip()
    else:
        message = "Validation error"

    body = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            # "Bad Request", the status phrase -- *not* "BadRequestError", which is
            # what the handler-code path uses.
            type=HTTPStatus.BAD_REQUEST.phrase,
            param=param,
            code=HTTPStatus.BAD_REQUEST.value,
        )
    )
    return JSONResponse(
        content=body.model_dump(), status_code=HTTPStatus.BAD_REQUEST.value
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Starlette's own errors -- a 404 on an unrouted path, a 405 -- reshaped.

    Without this they leave as `{"detail": "Not Found"}`, which is a different shape
    from every other error the server produces.
    """
    assert isinstance(exc, HTTPException)
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "Error"
    body = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(str(exc.detail)),
            type=phrase,
            code=exc.status_code,
        )
    )
    return JSONResponse(content=body.model_dump(), status_code=exc.status_code)


async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Anything that escaped a route, mapped by type rather than swallowed as a 500."""
    return to_error_response(exc)
