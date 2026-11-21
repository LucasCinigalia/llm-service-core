"""Adapter da Anthropic (Claude) para o port ``LLMProvider``.

É uma *anti-corruption layer*: tudo que é específico da API da Anthropic — formato das mensagens,
blocos de conteúdo, ``stop_reason``, contagem de tokens com cache, exceções do SDK — fica contido
aqui. Para fora saem apenas modelos de domínio (``ChatResponse``) e erros nossos (``core.errors``).

O arquivo se chama ``anthropic_provider.py`` (e não ``anthropic.py``) para não ser confundido
com o pacote ``anthropic`` do SDK.
"""

from collections.abc import AsyncGenerator

import anthropic
import httpx2
from anthropic.types.beta import (
    BetaAssistantMessageParam,
    BetaMessage,
    BetaMessageParam,
    BetaUserMessageParam,
)

from app.core.config import AnthropicEffort
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
    Role,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    Usage,
)

logger = get_logger(__name__)

# Fallback do lado do servidor: se o modelo recusar por política de segurança, a própria API
# refaz a chamada num modelo adequado ("default" escolhe pela categoria da recusa).
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    # A conversa encheu a janela de contexto: a resposta também foi cortada.
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.CONTENT_FILTER,
}


class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        effort: AnthropicEffort | None = None,
        refusal_fallback: bool = False,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._refusal_fallback = refusal_fallback
        # Um cliente por processo: ele mantém um pool de conexões HTTP reaproveitado entre requests.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "anthropic"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        max_tokens = self._validate(request)
        try:
            message = await self._client.beta.messages.create(
                model=request.model or self._model,
                max_tokens=max_tokens,
                system=request.system if request.system is not None else anthropic.omit,
                messages=[_to_message_param(m.role, m.content) for m in request.messages],
                betas=[REFUSAL_FALLBACK_BETA] if self._refusal_fallback else anthropic.omit,
                fallbacks="default" if self._refusal_fallback else anthropic.omit,
                output_config={"effort": self._effort} if self._effort else anthropic.omit,
            )
        except anthropic.APIError as exc:
            raise _translate_error(exc) from exc

        return self._to_domain(message)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        max_tokens = self._validate(request)
        try:
            # `async with` garante que a conexão é fechada mesmo se o consumidor parar no meio
            # (cliente desconectou): a Anthropic para de gerar — e de cobrar — os tokens.
            async with self._client.beta.messages.stream(
                model=request.model or self._model,
                max_tokens=max_tokens,
                system=request.system if request.system is not None else anthropic.omit,
                messages=[_to_message_param(m.role, m.content) for m in request.messages],
                betas=[REFUSAL_FALLBACK_BETA] if self._refusal_fallback else anthropic.omit,
                fallbacks="default" if self._refusal_fallback else anthropic.omit,
                output_config={"effort": self._effort} if self._effort else anthropic.omit,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield TextDelta(text=text)
                message = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise _translate_error(exc) from exc

        yield StreamCompleted(response=self._to_domain(message))

    def _validate(self, request: ChatRequest) -> int:
        """Validações comuns a ``chat`` e ``stream``; devolve o ``max_tokens`` a enviar."""
        if request.temperature is not None:
            # Os modelos Claude atuais não aceitam `temperature` (a API devolve 400). Ignoramos e
            # registramos, para o chamador saber que o parâmetro não teve efeito.
            logger.warning("llm_parameter_ignored", provider=self.name, parameter="temperature")
        if request.max_output_tokens is None:
            # O ChatService sempre preenche; a API da Anthropic exige o campo.
            msg = "max_output_tokens is required by the Anthropic API"
            raise LLMConfigurationError(msg)
        return request.max_output_tokens

    async def aclose(self) -> None:
        await self._client.close()

    def _to_domain(self, message: BetaMessage) -> ChatResponse:
        # A resposta é uma lista de blocos (thinking, text, fallback...). Só o texto vai ao usuário.
        content = "".join(block.text for block in message.content if block.type == "text")

        stop_reason = _STOP_REASONS.get(message.stop_reason or "")
        if stop_reason is None:
            # tool_use/pause_turn/compaction só ocorrem com recursos que ainda não usamos.
            msg = f"Unexpected stop_reason from Anthropic: {message.stop_reason!r}"
            raise LLMProviderError(msg)

        usage = message.usage
        return ChatResponse(
            content=content,
            model=message.model,  # pode ser o modelo de fallback, não o pedido
            provider=self.name,
            stop_reason=stop_reason,
            usage=Usage(
                # `input_tokens` da Anthropic exclui os tokens lidos/gravados no cache de prompt;
                # somamos para ter o total processado. O detalhamento de cache vem no commit 16.
                input_tokens=usage.input_tokens
                + (usage.cache_creation_input_tokens or 0)
                + (usage.cache_read_input_tokens or 0),
                output_tokens=usage.output_tokens,
            ),
        )


def _to_message_param(role: Role, content: str) -> BetaMessageParam:
    # O SDK tem um tipo por papel; o `match` exaustivo garante que um Role novo não passe batido.
    match role:
        case Role.USER:
            return BetaUserMessageParam(role="user", content=content)
        case Role.ASSISTANT:
            return BetaAssistantMessageParam(role="assistant", content=content)


def _translate_error(exc: anthropic.APIError) -> AppError:
    """Mapeia exceções do SDK para erros de domínio.

    A ordem dos ``case`` importa: do mais específico para o mais geral.
    """
    status_code = exc.status_code if isinstance(exc, anthropic.APIStatusError) else None
    provider_request_id = exc.request_id if isinstance(exc, anthropic.APIStatusError) else None
    # Detalhes do provedor vão para o log (diagnóstico), não para a resposta ao cliente.
    logger.warning(
        "llm_provider_error",
        provider="anthropic",
        error_type=type(exc).__name__,
        status_code=status_code,
        provider_request_id=provider_request_id,
        provider_message=exc.message,
    )
    details = {"provider": "anthropic", "provider_request_id": provider_request_id}

    error: AppError
    if status_code == 200:
        # Erro no MEIO do stream: o HTTP já tinha respondido 200, então o status não diz nada.
        # O tipo real do erro vem no corpo do evento SSE `error` (ex.: "overloaded_error").
        return _translate_stream_error(_body_error_type(exc), details)

    match exc:
        case anthropic.APITimeoutError():  # subclasse de APIConnectionError: vem antes
            error = LLMTimeoutError("The LLM provider timed out.", details=details)
        case anthropic.APIConnectionError():
            error = LLMUnavailableError("Could not reach the LLM provider.", details=details)
        case anthropic.RateLimitError():
            error = LLMRateLimitError("The LLM provider is rate limiting us.", details=details)
        case anthropic.InternalServerError():  # 500, 503, 504 e 529 (overloaded)
            error = LLMUnavailableError("The LLM provider is unavailable.", details=details)
        case (
            anthropic.AuthenticationError()
            | anthropic.PermissionDeniedError()
            | anthropic.NotFoundError()
        ):
            error = LLMConfigurationError("The LLM provider is misconfigured.", details=details)
        case (
            anthropic.BadRequestError()
            | anthropic.RequestTooLargeError()
            | anthropic.UnprocessableEntityError()
        ):
            error = LLMRequestRejectedError(
                "The LLM provider rejected the request.", details=details
            )
        case _:
            error = LLMProviderError("Unexpected LLM provider error.", details=details)
    return error


def _body_error_type(exc: anthropic.APIError) -> str | None:
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error_type = body["error"].get("type")
        return error_type if isinstance(error_type, str) else None
    return None


def _translate_stream_error(error_type: str | None, details: dict[str, str | None]) -> AppError:
    match error_type:
        case "rate_limit_error":
            return LLMRateLimitError("The LLM provider is rate limiting us.", details=details)
        case "overloaded_error" | "api_error" | "timeout_error":
            return LLMUnavailableError("The LLM provider is unavailable.", details=details)
        case _:
            return LLMProviderError("The LLM provider failed mid-stream.", details=details)
