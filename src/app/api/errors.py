"""Tradução de exceções em respostas HTTP com um formato único (envelope de erro).

Todo erro da API segue o mesmo contrato::

    {"error": {"code": "not_found", "message": "...", "request_id": "...", "details": {}}}
"""

from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError, ExternalServiceError, InvalidInputError, NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Mapeamento erro de domínio → status HTTP. Subclasses herdam o status da classe-mãe mais próxima.
STATUS_BY_ERROR: dict[type[AppError], HTTPStatus] = {
    NotFoundError: HTTPStatus.NOT_FOUND,
    InvalidInputError: HTTPStatus.UNPROCESSABLE_ENTITY,
    ExternalServiceError: HTTPStatus.BAD_GATEWAY,
    AppError: HTTPStatus.INTERNAL_SERVER_ERROR,
}


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


def error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request_id, details=details or {})
    )
    # int(): HTTPStatus é um IntEnum; sem a conversão, ele "vaza" para logs como
    # <HTTPStatus.NOT_FOUND: 404> em vez de 404.
    return JSONResponse(status_code=int(status_code), content=body.model_dump())


def status_for(exc: AppError) -> HTTPStatus:
    for cls in type(exc).__mro__:
        if cls in STATUS_BY_ERROR:
            return STATUS_BY_ERROR[cls]
    return HTTPStatus.INTERNAL_SERVER_ERROR


def _ensure_type[E: Exception](exc: Exception, expected: type[E]) -> E:
    """Estreita o tipo para o mypy. Os handlers recebem ``Exception`` porque é a assinatura que o
    Starlette exige; na prática o FastAPI só os chama com o tipo para o qual foram registrados."""
    if not isinstance(exc, expected):
        raise TypeError(f"expected {expected.__name__}, got {type(exc).__name__}")
    return exc


async def handle_app_error(_request: Request, exc: Exception) -> JSONResponse:
    exc = _ensure_type(exc, AppError)
    status = status_for(exc)
    log = logger.error if status >= HTTPStatus.INTERNAL_SERVER_ERROR else logger.warning
    log("app_error", code=exc.code, status_code=status.value, error_message=exc.message)
    return error_response(status, exc.code, exc.message, exc.details)


async def handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
    exc = _ensure_type(exc, StarletteHTTPException)
    status = HTTPStatus(exc.status_code)
    code = status.phrase.lower().replace(" ", "_").replace("-", "_")
    message = exc.detail if isinstance(exc.detail, str) else status.phrase
    return error_response(status, code, message)


async def handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    exc = _ensure_type(exc, RequestValidationError)
    # Removemos "input" (o valor recebido): ecoá-lo devolveria conteúdo possivelmente sensível
    # ou enorme (ex.: uma conversa de 100 mil caracteres). "loc" e "msg" bastam para corrigir.
    errors = [{k: v for k, v in error.items() if k != "input"} for error in exc.errors()]
    return error_response(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "validation_error",
        "Request validation failed.",
        {"errors": jsonable_encoder(errors)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
