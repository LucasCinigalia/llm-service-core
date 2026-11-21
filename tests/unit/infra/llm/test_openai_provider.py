"""Testes do adapter da OpenAI, sem rede e sem chave de API (ver ``tests/http_mock.py``)."""

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from app.core.errors import (
    AppError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.domain.llm import ChatRequest, Message, Role, StopReason
from app.domain.ports import LLMProvider
from app.infra.llm.openai_provider import OpenAIProvider
from http_mock import Handler, Recorder, json_response, raise_connection_error, raise_timeout

ProviderFactory = Callable[..., tuple[OpenAIProvider, Recorder]]


def completion_json(
    *, content: str | None = "Olá! Como posso ajudar?", **overrides: Any
) -> dict[str, Any]:
    """Uma resposta da API Chat Completions, no formato que a OpenAI devolve."""
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content, "refusal": None},
        "finish_reason": overrides.pop("finish_reason", "stop"),
    }
    if "refusal" in overrides:
        choice["message"]["refusal"] = overrides.pop("refusal")
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-test-2026-01-01",
        "choices": [choice],
        "usage": {"prompt_tokens": 18, "completion_tokens": 7, "total_tokens": 25},
        **overrides,
    }


def error_json(error_type: str, message: str = "boom") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}


@pytest.fixture
async def provider_factory() -> AsyncIterator[ProviderFactory]:
    created: list[OpenAIProvider] = []

    def factory(handler: Handler, **kwargs: Any) -> tuple[OpenAIProvider, Recorder]:
        recorder = Recorder(handler)
        provider = OpenAIProvider(
            api_key="sk-openai-test",
            model="gpt-test",
            timeout_seconds=5,
            max_retries=0,
            http_client=recorder.client(),
            **kwargs,
        )
        created.append(provider)
        return provider, recorder

    yield factory
    for provider in created:
        await provider.aclose()


def ok(**overrides: Any) -> Handler:
    return json_response(200, completion_json(**overrides))


def make_request(**kwargs: Any) -> ChatRequest:
    return ChatRequest(
        messages=(
            Message(role=Role.USER, content="Oi"),
            Message(role=Role.ASSISTANT, content="Olá!"),
            Message(role=Role.USER, content="Explique RAG."),
        ),
        max_output_tokens=kwargs.pop("max_output_tokens", 1000),
        **kwargs,
    )


async def test_satisfies_the_llm_provider_port(provider_factory: ProviderFactory) -> None:
    provider: LLMProvider
    provider, _ = provider_factory(ok())

    assert provider.name == "openai"


# --- Domínio → requisição da API --------------------------------------------------------------


async def test_translates_domain_request_into_chat_completions_call(
    provider_factory: ProviderFactory,
) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request(system="Seja conciso.", temperature=0.2))

    [http_request] = recorder.requests
    assert http_request.url.path == "/v1/chat/completions"
    assert http_request.headers["authorization"] == "Bearer sk-openai-test"
    assert recorder.last_body["model"] == "gpt-test"
    assert recorder.last_body["max_completion_tokens"] == 1000
    assert recorder.last_body["temperature"] == 0.2
    # O system prompt vira a PRIMEIRA mensagem — na Anthropic ele é um campo separado.
    assert recorder.last_body["messages"] == [
        {"role": "system", "content": "Seja conciso."},
        {"role": "user", "content": "Oi"},
        {"role": "assistant", "content": "Olá!"},
        {"role": "user", "content": "Explique RAG."},
    ]


async def test_omits_optional_fields_that_were_not_set(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request())

    body = recorder.last_body
    assert body["messages"][0]["role"] == "user"  # sem system prompt, sem mensagem de system
    assert "temperature" not in body
    assert "reasoning_effort" not in body


async def test_sends_configured_reasoning_effort(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok(), reasoning_effort="low")

    await provider.chat(make_request())

    assert recorder.last_body["reasoning_effort"] == "low"


async def test_custom_base_url_targets_openai_compatible_servers(
    provider_factory: ProviderFactory,
) -> None:
    # Ex.: Ollama, vLLM, Azure OpenAI — mesma API, outro endereço.
    provider, recorder = provider_factory(ok(), base_url="http://localhost:11434/v1")

    await provider.chat(make_request())

    assert str(recorder.requests[-1].url) == "http://localhost:11434/v1/chat/completions"


# --- Resposta da API → domínio --------------------------------------------------------------


async def test_translates_response_into_domain(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(ok())

    response = await provider.chat(make_request())

    assert response.content == "Olá! Como posso ajudar?"
    assert response.provider == "openai"
    assert response.model == "gpt-test-2026-01-01"  # o modelo exato que respondeu
    assert response.stop_reason is StopReason.END_TURN
    assert response.usage.input_tokens == 18
    assert response.usage.output_tokens == 7


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("content_filter", StopReason.CONTENT_FILTER),
    ],
)
async def test_maps_finish_reasons(
    provider_factory: ProviderFactory, finish_reason: str, expected: StopReason
) -> None:
    provider, _ = provider_factory(ok(finish_reason=finish_reason))

    response = await provider.chat(make_request())

    assert response.stop_reason is expected


async def test_explicit_refusal_maps_to_content_filter(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(ok(content=None, refusal="I can't help with that."))

    response = await provider.chat(make_request())

    assert response.stop_reason is StopReason.CONTENT_FILTER
    assert response.content == ""


async def test_unexpected_finish_reason_is_a_provider_error(
    provider_factory: ProviderFactory,
) -> None:
    provider, _ = provider_factory(ok(finish_reason="tool_calls"))

    with pytest.raises(LLMProviderError, match="tool_calls"):
        await provider.chat(make_request())


async def test_empty_choices_is_a_provider_error(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(ok(choices=[]))

    with pytest.raises(LLMProviderError, match="no choices"):
        await provider.chat(make_request())


async def test_missing_usage_is_tolerated(provider_factory: ProviderFactory) -> None:
    # Algumas APIs compatíveis com OpenAI não devolvem `usage`.
    provider, _ = provider_factory(ok(usage=None))

    response = await provider.chat(make_request())

    assert response.usage.total_tokens == 0


# --- Exceções do SDK → erros de domínio ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (429, "rate_limit_exceeded", LLMRateLimitError),
        (500, "server_error", LLMUnavailableError),
        (503, "server_error", LLMUnavailableError),
        (401, "invalid_api_key", LLMConfigurationError),
        (403, "permission_denied", LLMConfigurationError),
        (404, "model_not_found", LLMConfigurationError),
        (400, "invalid_request_error", LLMRequestRejectedError),
        (422, "invalid_request_error", LLMRequestRejectedError),
    ],
)
async def test_translates_http_errors(
    provider_factory: ProviderFactory,
    status: int,
    error_type: str,
    expected: type[AppError],
) -> None:
    provider, _ = provider_factory(
        json_response(status, error_json(error_type), **{"x-request-id": "req_openai_1"})
    )

    with pytest.raises(expected) as exc_info:
        await provider.chat(make_request())

    assert exc_info.value.details["provider_request_id"] == "req_openai_1"
    assert isinstance(exc_info.value.__cause__, Exception)


async def test_translates_timeouts(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(raise_timeout)

    with pytest.raises(LLMTimeoutError):
        await provider.chat(make_request())


async def test_translates_connection_errors(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(raise_connection_error)

    with pytest.raises(LLMUnavailableError):
        await provider.chat(make_request())
