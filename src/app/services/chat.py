"""Caso de uso: conversar com um LLM.

O service é onde fica a **regra de negócio**: aplicar padrões da configuração, registrar métricas,
reagir a respostas truncadas. Ele depende só do port ``LLMProvider`` — não sabe se do outro lado
está a Anthropic, a OpenAI ou o fake — e não sabe nada de HTTP.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing

from structlog.stdlib import BoundLogger

from app.core.config import LLMSettings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain.llm import (
    ChatRequest,
    ChatResponse,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
)
from app.domain.ports import LLMProvider

logger = get_logger(__name__)


class ChatService:
    def __init__(self, provider: LLMProvider, settings: LLMSettings) -> None:
        self._provider = provider
        self._settings = settings

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request = self._apply_defaults(request)
        log = logger.bind(provider=self._provider.name, requested_model=request.model)

        start = time.perf_counter()
        try:
            response = await self._provider.chat(request)
        except AppError as exc:
            log.warning(
                "llm_request_failed",
                error_code=exc.code,
                latency_ms=_elapsed_ms(start),
            )
            raise

        log.info(
            "llm_request_completed",
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=_elapsed_ms(start),
        )
        if response.stop_reason is StopReason.MAX_TOKENS:
            log.warning("llm_response_truncated", max_output_tokens=request.max_output_tokens)
        return response

    async def stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        """Como ``chat``, mas repassando os eventos conforme chegam do provider.

        Mede o **TTFT** (time to first token: quanto o usuário espera até ver algo) além da
        latência total, e registra se o stream terminou, falhou ou foi abandonado pelo cliente.
        """
        request = self._apply_defaults(request)
        log = logger.bind(provider=self._provider.name, requested_model=request.model)

        start = time.perf_counter()
        ttft_ms: float | None = None
        outcome = "cancelled"  # só muda se o stream terminar ou falhar
        try:
            # aclosing: se ESTE gerador for fechado no meio, fecha também o do provider —
            # e é isso que encerra a conexão HTTP e para a geração (e a cobrança) de tokens.
            async with aclosing(self._provider.stream(request)) as events:
                async for event in events:
                    if isinstance(event, TextDelta) and ttft_ms is None:
                        ttft_ms = _elapsed_ms(start)
                    if isinstance(event, StreamCompleted):
                        outcome = "completed"
                        self._log_stream_completed(log, event.response, request, start, ttft_ms)
                    yield event
        except AppError as exc:
            outcome = "failed"
            log.warning(
                "llm_stream_failed",
                error_code=exc.code,
                ttft_ms=ttft_ms,
                latency_ms=_elapsed_ms(start),
            )
            raise
        finally:
            if outcome == "cancelled":
                # O consumidor parou de ler antes do fim (ex.: o cliente HTTP desconectou).
                log.info("llm_stream_cancelled", ttft_ms=ttft_ms, latency_ms=_elapsed_ms(start))

    @staticmethod
    def _log_stream_completed(
        log: BoundLogger,
        response: ChatResponse,
        request: ChatRequest,
        start: float,
        ttft_ms: float | None,
    ) -> None:
        log.info(
            "llm_stream_completed",
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            ttft_ms=ttft_ms,
            latency_ms=_elapsed_ms(start),
        )
        if response.stop_reason is StopReason.MAX_TOKENS:
            log.warning("llm_response_truncated", max_output_tokens=request.max_output_tokens)

    def _apply_defaults(self, request: ChatRequest) -> ChatRequest:
        if request.max_output_tokens is not None:
            return request
        return request.model_copy(update={"max_output_tokens": self._settings.max_output_tokens})


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
