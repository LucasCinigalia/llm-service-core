"""Middleware de contexto da request: correlation ID, log de acesso e erros inesperados.

É um middleware **ASGI puro** (e não ``BaseHTTPMiddleware``) porque:
- não bufferiza o corpo da resposta — essencial para streaming (SSE) no commit 12;
- roda no mesmo contexto da request, então as ``contextvars`` funcionam de ponta a ponta.
"""

import re
import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response
from app.core.logging import get_logger

REQUEST_ID_HEADER = "X-Request-ID"
# Aceitamos o ID do cliente só se for "seguro": evita log injection e valores gigantes.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")

logger = get_logger(__name__)


def resolve_request_id(incoming: str | None) -> str:
    if incoming and _VALID_REQUEST_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # ex.: eventos de lifespan, websockets
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status_code = 500
        response_started = False
        start = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception("unhandled_exception")
            if response_started:
                # Parte da resposta já foi enviada; não dá para trocar por um erro JSON.
                raise
            response = error_response(500, "internal_error", "An unexpected error occurred.")
            await response(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "http_request_completed",
                method=scope["method"],
                path=scope["path"],  # sem query string: pode conter dados sensíveis
                status_code=status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()
