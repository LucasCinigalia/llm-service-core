"""Rota de chat com streaming via Server-Sent Events (SSE).

Protocolo (``Content-Type: text/event-stream``)::

    event: delta
    data: {"text": "Olá, "}

    event: delta
    data: {"text": "tudo bem?"}

    event: done
    data: {"content": "Olá, tudo bem?", "model": "...", "usage": {...}, ...}

Se algo falhar **depois** que o stream começou, o status 200 já foi enviado — então o erro vem
como um evento ``error`` (mesmo formato do envelope de erro da API) e o stream termina::

    event: error
    data: {"code": "llm_unavailable", "message": "...", "request_id": "..."}

Erros **antes** do primeiro evento (validação, provider fora do ar, rate limit) continuam sendo
respostas HTTP normais (422, 503...) com o envelope JSON padrão.
"""

from collections.abc import AsyncGenerator
from contextlib import aclosing

import structlog
from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, format_sse_event
from pydantic import BaseModel

from app.api.dependencies import ChatServiceDep
from app.api.errors import ErrorBody, ErrorResponse
from app.api.schemas.chat import ChatRequestBody, ChatResponseBody, StreamDeltaOut
from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain.llm import StreamCompleted, StreamEvent, TextDelta

router = APIRouter(prefix="/v1", tags=["chat"])
logger = get_logger(__name__)

SSE_HEADERS = {
    # Ninguém no caminho (navegador, CDN, proxy) deve guardar ou juntar os eventos.
    "Cache-Control": "no-cache",
    # Desliga o buffer do nginx: sem isto, ele acumula os eventos e o streaming "trava".
    "X-Accel-Buffering": "no",
}


# Sem `response_class=EventSourceResponse`: no FastAPI esse marcador significa "a rota é um
# gerador SSE". Aqui a rota é uma função normal que lê o 1º evento e RETORNA a resposta pronta.
@router.post(
    "/chat/stream",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Eventos SSE: `delta`*, depois `done` (ou `error`).",
        },
        422: {"model": ErrorResponse, "description": "Requisição inválida"},
        503: {"model": ErrorResponse, "description": "Provedor de LLM indisponível"},
    },
)
async def chat_stream(body: ChatRequestBody, service: ChatServiceDep) -> EventSourceResponse:
    """Envia uma conversa ao LLM e devolve a resposta aos poucos, via Server-Sent Events."""
    events = service.stream(body.to_domain())
    try:
        # Lemos o 1º evento ANTES de responder: se o provider falhar logo de cara, a exceção
        # sobe daqui e vira uma resposta HTTP de erro normal (o status ainda não foi enviado).
        first = await anext(events)
    except BaseException:
        await events.aclose()
        raise
    return EventSourceResponse(_encode(first, events), headers=SSE_HEADERS)


async def _encode(
    first: StreamEvent, events: AsyncGenerator[StreamEvent, None]
) -> AsyncGenerator[bytes, None]:
    """Converte eventos de domínio em bytes SSE. Erros no meio viram um evento ``error``."""
    async with aclosing(events):
        try:
            yield _to_sse(first)
            async for event in events:
                yield _to_sse(event)
        except AppError as exc:
            yield _error_event(exc.code, exc.message)
        except Exception:
            # Bug nosso no meio do stream: loga com traceback, mas não vaza detalhes ao cliente.
            logger.exception("unhandled_stream_exception")
            yield _error_event("internal_error", "An unexpected error occurred.")


def _to_sse(event: StreamEvent) -> bytes:
    match event:
        case TextDelta():
            return _sse("delta", StreamDeltaOut(text=event.text))
        case StreamCompleted():
            return _sse("done", ChatResponseBody.from_domain(event.response))


def _error_event(code: str, message: str) -> bytes:
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    return _sse("error", ErrorBody(code=code, message=message, request_id=request_id))


def _sse(event: str, data: BaseModel) -> bytes:
    return format_sse_event(event=event, data_str=data.model_dump_json())
