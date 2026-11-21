"""Testes do adapter da Anthropic, sem rede e sem chave de API.

Injetamos no SDK um cliente ``httpx2`` com ``MockTransport``: o SDK monta a requisição HTTP real
(URL, headers, JSON) e, em vez de mandá-la à internet, ela vai para a função ``handler`` do teste.
Assim testamos o adapter *e* a integração com o SDK de verdade — só a rede é simulada.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
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
from app.domain.llm import ChatRequest, Message, Role, StopReason, StreamCompleted, TextDelta
from app.domain.ports import LLMProvider
from app.infra.llm.anthropic_provider import REFUSAL_FALLBACK_BETA, AnthropicProvider
from http_mock import Handler, Recorder, raise_connection_error, raise_timeout, sse_response
from stream_helpers import drain_texts


def message_json(**overrides: Any) -> dict[str, Any]:
    """Uma resposta da API de Messages, no formato que a Anthropic devolve."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Olá! "},
            {"type": "text", "text": "Como posso ajudar?"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 7,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 5,
        },
        **overrides,
    }


def error_json(error_type: str, message: str = "boom") -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


ProviderFactory = Callable[..., tuple[AnthropicProvider, Recorder]]


@pytest.fixture
async def provider_factory() -> AsyncIterator[ProviderFactory]:
    """Cria providers ligados a um ``handler`` falso e fecha todos ao fim do teste."""
    created: list[AnthropicProvider] = []

    def factory(handler: Handler, **kwargs: Any) -> tuple[AnthropicProvider, Recorder]:
        recorder = Recorder(handler)
        provider = AnthropicProvider(
            api_key="sk-ant-test",
            model="claude-haiku-5-5",
            timeout_seconds=5,
            max_retries=0,  # sem retentativas do SDK: cada teste vê exatamente uma chamada
            http_client=recorder.client(),
            **kwargs,
        )
        created.append(provider)
        return provider, recorder

    yield factory
    for provider in created:
        await provider.aclose()


def ok(**overrides: Any) -> Handler:
    return lambda _request: httpx2.Response(200, json=message_json(**overrides))


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

    assert provider.name == "anthropic"


# --- Domínio → requisição da API --------------------------------------------------------------


async def test_translates_domain_request_into_messages_api_call(
    provider_factory: ProviderFactory,
) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request(system="Seja conciso."))

    [http_request] = recorder.requests
    assert http_request.method == "POST"
    assert http_request.url.path == "/v1/messages"
    assert http_request.headers["authorization"] == "Bearer sk-ant-test"
    assert recorder.last_body["model"] == "claude-haiku-5-5"
    assert recorder.last_body["max_tokens"] == 1000
    assert recorder.last_body["system"] == "Seja conciso."
    assert recorder.last_body["messages"] == [
        {"role": "user", "content": "Oi"},
        {"role": "assistant", "content": "Olá!"},
        {"role": "user", "content": "Explique RAG."},
    ]


async def test_omits_optional_fields_that_were_not_set(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request())

    assert "system" not in recorder.last_body


async def test_requested_model_overrides_default(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request(model="claude-sonnet-5-5"))

    assert recorder.last_body["model"] == "claude-sonnet-5-5"


async def test_does_not_send_temperature(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request(temperature=0.2))

    assert "temperature" not in recorder.last_body


async def test_refusal_fallback_is_off_by_default(provider_factory: ProviderFactory) -> None:
    # O padrão é o Haiku 5.5, que não tem fallback de recusa no servidor.
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request())

    assert "fallbacks" not in recorder.last_body
    assert "anthropic-beta" not in recorder.requests[-1].headers


async def test_refusal_fallback_can_be_enabled(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok(), refusal_fallback=True)

    await provider.chat(make_request(model="claude-opus-5-5"))

    assert recorder.last_body["fallbacks"] == "default"
    assert REFUSAL_FALLBACK_BETA in recorder.requests[-1].headers["anthropic-beta"]


async def test_uses_model_default_effort_when_not_configured(
    provider_factory: ProviderFactory,
) -> None:
    provider, recorder = provider_factory(ok())

    await provider.chat(make_request())

    assert "output_config" not in recorder.last_body


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_sends_configured_effort(provider_factory: ProviderFactory, effort: str) -> None:
    provider, recorder = provider_factory(ok(), effort=effort)

    await provider.chat(make_request())

    assert recorder.last_body["output_config"] == {"effort": effort}


async def test_requires_max_output_tokens(provider_factory: ProviderFactory) -> None:
    provider, recorder = provider_factory(ok())
    request = ChatRequest(messages=(Message(role=Role.USER, content="oi"),))

    with pytest.raises(LLMConfigurationError, match="max_output_tokens"):
        await provider.chat(request)
    assert recorder.requests == []


# --- Resposta da API → domínio --------------------------------------------------------------


async def test_translates_response_into_domain(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(ok())

    response = await provider.chat(make_request())

    assert response.content == "Olá! Como posso ajudar?"  # só blocos de texto, concatenados
    assert response.provider == "anthropic"
    assert response.model == "claude-haiku-5-5"
    assert response.stop_reason is StopReason.END_TURN


async def test_input_tokens_include_prompt_cache_tokens(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(ok())

    response = await provider.chat(make_request())

    assert response.usage.input_tokens == 10 + 3 + 5
    assert response.usage.output_tokens == 7


async def test_reports_the_model_that_actually_answered(provider_factory: ProviderFactory) -> None:
    # Com fallback, quem responde pode ser outro modelo.
    provider, _ = provider_factory(ok(model="claude-opus-4-8"))

    response = await provider.chat(make_request())

    assert response.model == "claude-opus-4-8"


@pytest.mark.parametrize(
    ("api_stop_reason", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("model_context_window_exceeded", StopReason.MAX_TOKENS),
        ("stop_sequence", StopReason.STOP_SEQUENCE),
        ("refusal", StopReason.CONTENT_FILTER),
    ],
)
async def test_maps_stop_reasons(
    provider_factory: ProviderFactory, api_stop_reason: str, expected: StopReason
) -> None:
    provider, _ = provider_factory(ok(stop_reason=api_stop_reason))

    response = await provider.chat(make_request())

    assert response.stop_reason is expected


async def test_refusal_may_have_no_text(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(ok(stop_reason="refusal", content=[]))

    response = await provider.chat(make_request())

    assert response.content == ""
    assert response.stop_reason is StopReason.CONTENT_FILTER


async def test_unexpected_stop_reason_is_a_provider_error(
    provider_factory: ProviderFactory,
) -> None:
    provider, _ = provider_factory(ok(stop_reason="tool_use"))

    with pytest.raises(LLMProviderError, match="tool_use"):
        await provider.chat(make_request())


# --- Exceções do SDK → erros de domínio ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (429, "rate_limit_error", LLMRateLimitError),
        (529, "overloaded_error", LLMUnavailableError),
        (500, "api_error", LLMUnavailableError),
        (401, "authentication_error", LLMConfigurationError),
        (403, "permission_error", LLMConfigurationError),
        (404, "not_found_error", LLMConfigurationError),
        (400, "invalid_request_error", LLMRequestRejectedError),
        (413, "request_too_large", LLMRequestRejectedError),
    ],
)
async def test_translates_http_errors(
    provider_factory: ProviderFactory,
    status: int,
    error_type: str,
    expected: type[AppError],
) -> None:
    provider, _ = provider_factory(
        lambda _r: httpx2.Response(
            status, json=error_json(error_type), headers={"request-id": "req_abc"}
        )
    )

    with pytest.raises(expected) as exc_info:
        await provider.chat(make_request())

    assert exc_info.value.details["provider_request_id"] == "req_abc"
    assert isinstance(exc_info.value.__cause__, Exception)  # exceção original preservada


async def test_translates_timeouts(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(raise_timeout)

    with pytest.raises(LLMTimeoutError):
        await provider.chat(make_request())


async def test_translates_connection_errors(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(raise_connection_error)

    with pytest.raises(LLMUnavailableError):
        await provider.chat(make_request())


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (LLMRateLimitError, True),
        (LLMUnavailableError, True),
        (LLMTimeoutError, True),
        (LLMRequestRejectedError, False),
        (LLMConfigurationError, False),
    ],
)
def test_transient_errors_are_marked_retryable(error: type[AppError], retryable: bool) -> None:
    assert error("x").retryable is retryable


# --- Streaming --------------------------------------------------------------------------------


def stream_events(*texts: str, stop_reason: str = "end_turn") -> list[tuple[str | None, Any]]:
    """Os eventos SSE que a API de Messages envia num streaming de texto."""
    events: list[tuple[str | None, Any]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **message_json(content=[], stop_reason=None),
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    events += [
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}},
        )
        for t in texts
    ]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


async def test_stream_yields_text_deltas_then_completed_response(
    provider_factory: ProviderFactory,
) -> None:
    provider, recorder = provider_factory(sse_response(stream_events("Olá", "! Tudo", " bem?")))

    events = [event async for event in provider.stream(make_request())]

    *deltas, completed = events
    assert [d.text for d in deltas if isinstance(d, TextDelta)] == ["Olá", "! Tudo", " bem?"]
    assert isinstance(completed, StreamCompleted)
    assert completed.response.content == "Olá! Tudo bem?"
    assert completed.response.stop_reason is StopReason.END_TURN
    assert completed.response.usage.output_tokens == 6
    assert recorder.last_body["stream"] is True


async def test_stream_maps_final_stop_reason(provider_factory: ProviderFactory) -> None:
    provider, _ = provider_factory(sse_response(stream_events("corta", stop_reason="max_tokens")))

    events = [event async for event in provider.stream(make_request())]

    assert isinstance(events[-1], StreamCompleted)
    assert events[-1].response.stop_reason is StopReason.MAX_TOKENS


async def test_stream_errors_before_first_event_are_translated(
    provider_factory: ProviderFactory,
) -> None:
    provider, _ = provider_factory(
        lambda _r: httpx2.Response(429, json=error_json("rate_limit_error"))
    )

    with pytest.raises(LLMRateLimitError):
        _ = [event async for event in provider.stream(make_request())]


async def test_mid_stream_error_is_translated_by_its_body_type(
    provider_factory: ProviderFactory,
) -> None:
    # O HTTP já respondeu 200; o erro chega como um evento SSE `error` no meio do texto.
    events = [*stream_events("Olá")[:3], ("error", error_json("overloaded_error"))]
    provider, _ = provider_factory(sse_response(events))
    received: list[str] = []

    with pytest.raises(LLMUnavailableError) as exc_info:
        await drain_texts(provider.stream(make_request()), into=received)

    assert received == ["Olá"]
    assert exc_info.value.retryable is True
