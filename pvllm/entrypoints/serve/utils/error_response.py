"""Error responses.

Upstream: vllm/entrypoints/serve/utils/error_response.py
Tier: C

C7 binds error codes and failure modes at capacity, and R2.5 binds the specific
errors. The shape is OpenAI's, because a client's error handling is written against
that -- a differently-shaped error is as breaking as a wrong status code.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pvllm.entrypoints.serve.utils.api_utils import sanitize_message


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int | str | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> JSONResponse:
    """Every error body the server builds passes through here -- and therefore through
    `sanitize_message`, as upstream does it.

    Sanitising at this choke point rather than at the handlers is the whole point.
    `to_error_response` feeds this `str(exc)` four times over, and an arbitrary
    exception's `str` is precisely the message that carries a traceback frame or an
    absolute path. Wiring the sanitiser into the two exception handlers only, as M8
    first did, left it inert for every error raised inside handler code -- which is
    most of them.
    """
    return JSONResponse(
        status_code=status_code.value,
        content=ErrorResponse(
            error=ErrorInfo(
                message=sanitize_message(message),
                type=err_type,
                param=param,
                code=status_code.value,
            )
        ).model_dump(),
    )


def model_not_found(model: str, served: list[str]) -> JSONResponse:
    """R2.5. 404 with the served names, so a misconfigured client can self-diagnose."""
    return create_error_response(
        message=(
            f"The model `{model}` does not exist. Served models: {sorted(served)}."
        ),
        err_type="NotFoundError",
        status_code=HTTPStatus.NOT_FOUND,
        param="model",
    )


def not_implemented(message: str, param: str | None = None) -> JSONResponse:
    """A feature this build does not model, refused by name. C7.

    One helper so that a refusal *returned* from a handler and a `NotImplementedError`
    *raised* from one produce the same status, and so that moving that status again
    means editing one line rather than ten call sites.

    An earlier version of this docstring said the two paths "used to disagree -- 400
    from the former, 501 from the latter". They did not: before M8 both were 400, and
    501 appears nowhere earlier in the history. The helper unifies them at a *new*
    status; it did not repair a split.
    """
    return create_error_response(
        message,
        err_type="NotImplementedError",
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        param=param,
    )


def to_error_response(exc: Exception) -> JSONResponse:
    """Map an exception onto the status and type upstream would give it. C7.

    `NotImplementedError` is a **501**, matching upstream's own mapping. It used to be
    a 400 here, with a docstring arguing that a feature this build does not model is
    the client's problem rather than the server's -- but that argument was against
    reporting *500*, and 501 answers it better than 400 did: it says precisely "this
    server does not implement that", it is what a real vLLM returns for the same
    exception, and no sensible client retries it.

    `ValueError`, `TypeError` and `OverflowError` are the client's problem and become
    400. A jinja2 `TemplateError` is too -- it means the chat template rejected the
    messages -- and is matched by name so this module need not import jinja2.
    """
    if isinstance(exc, NotImplementedError):
        return create_error_response(
            str(exc),
            err_type="NotImplementedError",
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )
    if isinstance(exc, ValueError | TypeError | OverflowError):
        return create_error_response(str(exc))
    if any(cls.__name__ == "TemplateError" for cls in type(exc).__mro__):
        return create_error_response(str(exc))
    return create_error_response(
        str(exc),
        err_type="InternalServerError",
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def as_dict(response: JSONResponse) -> dict[str, Any]:
    import json

    return dict(json.loads(bytes(response.body)))
