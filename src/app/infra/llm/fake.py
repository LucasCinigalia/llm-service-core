"""Provider falso: determinístico, instantâneo e gratuito.

Serve para dois cenários:
- **Desenvolvimento local** sem chave de API (``LLM__DEFAULT_PROVIDER=fake``): responde com um eco.
- **Testes**: você programa as respostas (ou erros) e depois inspeciona as requests recebidas.

Uso em testes::

    provider = FakeLLMProvider(["primeira resposta", ExternalServiceError("fora do ar")])
    await provider.chat(request)       # -> "primeira resposta"
    await provider.chat(request)       # -> levanta ExternalServiceError
    assert provider.requests[0] == request
"""

import asyncio
from collections import deque
from collections.abc import Iterable

from app.domain.llm import ChatRequest, ChatResponse, StopReason, Usage

# Uma resposta programada pode ser só o texto, uma resposta completa ou um erro a ser lançado.
ScriptedItem = str | ChatResponse | Exception

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
    ) -> None:
        self._name = name
        self._model = model
        self._latency_seconds = latency_seconds
        self._scripted: deque[ScriptedItem] = deque(responses)
        self.requests: list[ChatRequest] = []

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

        item = self._scripted.popleft() if self._scripted else self._echo(request)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ChatResponse):
            return item
        return self._build_response(request, item)

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
