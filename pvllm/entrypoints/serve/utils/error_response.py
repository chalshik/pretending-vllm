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
    return JSONResponse(
        status_code=status_code.value,
        content=ErrorResponse(
            error=ErrorInfo(
                message=message,
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


def to_error_response(exc: Exception) -> JSONResponse:
    """Map an engine exception onto the right status.

    `NotImplementedError` becomes a 400 rather than a 500 on purpose: it means the
    client asked for a feature this build does not model, which is a problem with the
    request, not with the server. Reporting 500 would send a client into retry
    logic for something that will never succeed.
    """
    if isinstance(exc, NotImplementedError):
        return create_error_response(
            str(exc), err_type="NotImplementedError", status_code=HTTPStatus.BAD_REQUEST
        )
    if isinstance(exc, ValueError):
        return create_error_response(str(exc))
    return create_error_response(
        str(exc),
        err_type="InternalServerError",
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def as_dict(response: JSONResponse) -> dict[str, Any]:
    import json

    return dict(json.loads(bytes(response.body)))
