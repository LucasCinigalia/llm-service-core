"""Adapter da OpenAI (Chat Completions) para o port ``LLMProvider``.

Usamos a API **Chat Completions** (e não a Responses API, mais nova) por um motivo prático: ela é
o padrão "OpenAI-compatible" que quase todo servidor de LLM implementa — Azure OpenAI, vLLM,
Ollama, LM Studio, Groq. Com ``base_url`` configurável, este único adapter atende todos eles.

Diferenças em relação à Anthropic que ficam contidas aqui:
- o system prompt é uma *mensagem* (``role: "system"``), não um campo separado;
- o limite de saída se chama ``max_completion_tokens``;
- o motivo de parada é ``finish_reason`` e tem outros valores;
- a resposta vem em ``choices[0].message`` e os tokens em ``prompt_tokens``/``completion_tokens``.
"""

from collections.abc import AsyncGenerator

import httpx2
import openai
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.core.config import OpenAIReasoningEffort
from app.core.errors import (
    AppError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logging import get_logger
from app.domain.llm import (
    ChatRequest,
    ChatResponse,
    Message,
    Role,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    Usage,
)

logger = get_logger(__name__)

_FINISH_REASONS: dict[str, StopReason] = {
    # A OpenAI usa "stop" tanto para fim natural quanto para stop sequence: a distinção se perde.
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.CONTENT_FILTER,
}


class OpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        base_url: str | None = None,
        reasoning_effort: OpenAIReasoningEffort | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "openai"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            completion = await self._client.chat.completions.create(
                model=request.model or self._model,
                messages=_to_messages(request),
                max_completion_tokens=(
                    request.max_output_tokens
                    if request.max_output_tokens is not None
                    else openai.omit
                ),
                temperature=(
                    request.temperature if request.temperature is not None else openai.omit
                ),
                reasoning_effort=self._reasoning_effort or openai.omit,
            )
        except openai.APIError as exc:
            raise _translate_error(exc) from exc

        return self._to_domain(completion)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        parts: list[str] = []
        model = request.model or self._model
        finish_reason: str | None = None
        refused = False
        usage: CompletionUsage | None = None
        try:
            chunks = await self._client.chat.completions.create(
                model=request.model or self._model,
                messages=_to_messages(request),
                max_completion_tokens=(
                    request.max_output_tokens
                    if request.max_output_tokens is not None
                    else openai.omit
                ),
                temperature=(
                    request.temperature if request.temperature is not None else openai.omit
                ),
                reasoning_effort=self._reasoning_effort or openai.omit,
                stream=True,
                # Sem isto a OpenAI não informa os tokens no streaming. Eles chegam num último
                # chunk, sem `choices`.
                stream_options={"include_usage": True},
            )
            # `async with` fecha a conexão se o consumidor parar no meio (cliente desconectou).
            async with chunks:
                async for chunk in chunks:
                    model = chunk.model or model
                    usage = chunk.usage or usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    refused = refused or bool(choice.delta.refusal)
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                        yield TextDelta(text=choice.delta.content)
        except openai.APIError as exc:
            raise _translate_error(exc) from exc

        yield StreamCompleted(
            response=self._build_response(
                content="".join(parts),
                model=model,
                finish_reason=finish_reason,
                refused=refused,
                usage=usage,
            )
        )

    async def aclose(self) -> None:
        await self._client.close()

    def _to_domain(self, completion: ChatCompletion) -> ChatResponse:
        if not completion.choices:
            msg = "OpenAI returned no choices"
            raise LLMProviderError(msg)
        choice = completion.choices[0]
        return self._build_response(
            content=choice.message.content or "",
            model=completion.model,
            finish_reason=choice.finish_reason,
            refused=bool(choice.message.refusal),
            usage=completion.usage,
        )

    def _build_response(
        self,
        *,
        content: str,
        model: str,
        finish_reason: str | None,
        refused: bool,
        usage: CompletionUsage | None,
    ) -> ChatResponse:
        """Monta a resposta de domínio — compartilhado por ``chat`` e ``stream``."""
        if refused:
            # Recusa explícita do modelo (campo próprio, separado do conteúdo).
            stop_reason = StopReason.CONTENT_FILTER
        else:
            mapped = _FINISH_REASONS.get(finish_reason or "")
            if mapped is None:
                # tool_calls/function_call só ocorrem com tools, que ainda não usamos.
                msg = f"Unexpected finish_reason from OpenAI: {finish_reason!r}"
                raise LLMProviderError(msg)
            stop_reason = mapped

        if usage is None:
            # Algumas APIs compatíveis omitem `usage`. Não falhamos, mas registramos.
            logger.warning("llm_usage_missing", provider=self.name, model=model)

        return ChatResponse(
            content=content,
            model=model,
            provider=self.name,
            stop_reason=stop_reason,
            usage=Usage(
                # Na OpenAI, `prompt_tokens` já inclui os tokens em cache e `completion_tokens`
                # já inclui os de raciocínio — o oposto da convenção da Anthropic.
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )


def _to_messages(request: ChatRequest) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    if request.system is not None:
        messages.append(ChatCompletionSystemMessageParam(role="system", content=request.system))
    messages.extend(_to_message_param(message) for message in request.messages)
    return messages


def _to_message_param(message: Message) -> ChatCompletionMessageParam:
    match message.role:
        case Role.USER:
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case Role.ASSISTANT:
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)


def _translate_error(exc: openai.APIError) -> AppError:
    """Mapeia exceções do SDK para erros de domínio.

    A ordem dos ``case`` importa: do mais específico para o mais geral.
    """
    status_code = exc.status_code if isinstance(exc, openai.APIStatusError) else None
    provider_request_id = exc.request_id if isinstance(exc, openai.APIStatusError) else None
    logger.warning(
        "llm_provider_error",
        provider="openai",
        error_type=type(exc).__name__,
        status_code=status_code,
        provider_request_id=provider_request_id,
        provider_message=exc.message,
    )
    details = {"provider": "openai", "provider_request_id": provider_request_id}

    error: AppError
    match exc:
        case openai.APITimeoutError():  # subclasse de APIConnectionError: vem antes
            error = LLMTimeoutError("The LLM provider timed out.", details=details)
        case openai.APIConnectionError():
            error = LLMUnavailableError("Could not reach the LLM provider.", details=details)
        case openai.RateLimitError():
            error = LLMRateLimitError("The LLM provider is rate limiting us.", details=details)
        case openai.InternalServerError():
            error = LLMUnavailableError("The LLM provider is unavailable.", details=details)
        case openai.AuthenticationError() | openai.PermissionDeniedError() | openai.NotFoundError():
            error = LLMConfigurationError("The LLM provider is misconfigured.", details=details)
        case openai.BadRequestError() | openai.UnprocessableEntityError():
            error = LLMRequestRejectedError(
                "The LLM provider rejected the request.", details=details
            )
        case _:
            error = LLMProviderError("Unexpected LLM provider error.", details=details)
    return error
