"""Provider falso: determinístico, instantâneo e gratuito.

Serve para dois cenários:
- **Desenvolvimento local** sem chave de API (``LLM__DEFAULT_PROVIDER=fake``): responde com um eco.
- **Testes**: você programa as respostas (ou erros) e depois inspeciona as requests recebidas.

Uso em testes::

    provider = FakeLLMProvider(["primeira resposta", ExternalServiceError("fora do ar")])
    await provider.chat(request)       # -> "primeira resposta"
    await provider.chat(request)       # -> levanta ExternalServiceError
    assert provider.requests[0] == request

No streaming, o texto é entregue palavra por palavra (``TextDelta``) e, por fim, um
``StreamCompleted``. ``FailMidStream`` simula uma falha depois que parte do texto já saiu.
"""

import asyncio
import re
from collections import deque
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass

from app.domain.llm import (
    ChatRequest,
    ChatResponse,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    Usage,
)


@dataclass(frozen=True)
class FailMidStream:
    """Programa uma falha no meio do stream: envia ``after_chunks`` pedaços e depois ``error``.

    Em ``chat()`` (sem streaming), equivale a lançar ``error`` direto.
    """

    error: Exception
    after_chunks: int = 1
    content: str = "resposta que vai falhar no meio"


# Uma resposta programada pode ser o texto, uma resposta completa, um erro ou uma falha no meio.
ScriptedItem = str | ChatResponse | Exception | FailMidStream

# Regra prática: ~4 caracteres por token em inglês (em português costuma ser um pouco menos).
# Para o fake basta uma estimativa estável; providers reais devolvem a contagem exata.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class FakeLLMProvider:
    def __init__(
        self,
        responses: Iterable[ScriptedItem] = (),
        *,
        name: str = "fake",
        model: str = "fake-model",
        latency_seconds: float = 0.0,
        chunk_delay_seconds: float = 0.0,
    ) -> None:
        self._name = name
        self._model = model
        self._latency_seconds = latency_seconds  # antes da resposta / do 1º pedaço (TTFT)
        self._chunk_delay_seconds = chunk_delay_seconds  # entre pedaços do stream
        self._scripted: deque[ScriptedItem] = deque(responses)
        self.requests: list[ChatRequest] = []
        self.aborted_streams = 0  # streams interrompidos pelo consumidor antes do fim

    @property
    def name(self) -> str:
        return self._name

    def enqueue(self, *items: ScriptedItem) -> None:
        """Programa as próximas respostas (consumidas na ordem, uma por chamada)."""
        self._scripted.extend(items)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)

        item = self._next_item(request)
        if isinstance(item, FailMidStream):
            raise item.error
        return item

    async def stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        self.requests.append(request)
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)

        item = self._next_item(request)
        response = (
            self._build_response(request, item.content) if isinstance(item, FailMidStream) else item
        )
        completed = False
        try:
            for index, chunk in enumerate(_split_into_chunks(response.content)):
                if isinstance(item, FailMidStream) and index == item.after_chunks:
                    raise item.error
                if index and self._chunk_delay_seconds:
                    await asyncio.sleep(self._chunk_delay_seconds)
                yield TextDelta(text=chunk)
            yield StreamCompleted(response=response)
            completed = True
        finally:
            # Roda também quando o consumidor para de iterar (aclose/cancelamento).
            if not completed:
                self.aborted_streams += 1

    def _next_item(self, request: ChatRequest) -> ChatResponse | FailMidStream:
        item = self._scripted.popleft() if self._scripted else self._echo(request)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return self._build_response(request, item)
        return item

    async def aclose(self) -> None:
        """Nada a liberar: o fake não abre conexões."""

    @staticmethod
    def _echo(request: ChatRequest) -> str:
        return f"[fake] {request.messages[-1].content}"

    def _build_response(self, request: ChatRequest, content: str) -> ChatResponse:
        stop_reason = StopReason.END_TURN
        limit = request.max_output_tokens
        if limit is not None and estimate_tokens(content) > limit:
            # Simula o corte por limite de tokens, como um provider real faria.
            content = content[: limit * CHARS_PER_TOKEN]
            stop_reason = StopReason.MAX_TOKENS

        prompt = " ".join([request.system or "", *(m.content for m in request.messages)])
        return ChatResponse(
            content=content,
            model=request.model or self._model,
            provider=self._name,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(content),
            ),
        )


def _split_into_chunks(text: str) -> list[str]:
    """Divide em pedaços de "palavra + espaço" — a concatenação devolve o texto original."""
    return re.findall(r"\S+\s*|\s+", text)
