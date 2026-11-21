"""Testes de contrato do port ``LLMProvider``.

A *mesma* suíte roda contra todos os adapters (fake, anthropic, openai). O port diz quais métodos
existem (o mypy verifica); o contrato diz como eles devem se **comportar** — e isso o mypy não vê.

Para adicionar um provider novo: implemente o adapter e um ``Harness`` aqui. Se a suíte passar,
ele é intercambiável com os outros do ponto de vista da aplicação.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar, Literal, Protocol

import pytest

from app.core.errors import LLMRateLimitError, LLMTimeoutError
from app.domain.llm import (
    ChatRequest,
    ChatResponse,
    Message,
    Role,
    StopReason,
    StreamCompleted,
    TextDelta,
)
from app.domain.ports import LLMProvider
from app.infra.llm.anthropic_provider import AnthropicProvider
from app.infra.llm.fake import FakeLLMProvider
from app.infra.llm.openai_provider import OpenAIProvider
from http_mock import Handler, Recorder, json_response, raise_timeout, sse_response

Scenario = Literal["ok", "stream_ok", "rate_limited", "timeout"]


class Harness(Protocol):
    """Sabe criar um provider já programado para responder um cenário."""

    def build(self, scenario: Scenario) -> LLMProvider: ...


class FakeHarness:
    def build(self, scenario: Scenario) -> LLMProvider:
        match scenario:
            case "ok" | "stream_ok":
                return FakeLLMProvider(["Olá, tudo bem?"])
            case "rate_limited":
                return FakeLLMProvider([LLMRateLimitError("rate limited")])
            case "timeout":
                return FakeLLMProvider([LLMTimeoutError("timed out")])


class AnthropicHarness:
    # ClassVar: constante da classe, não um atributo de instância (evita estado compartilhado).
    responses: ClassVar[dict[Scenario, Handler]] = {
        "ok": json_response(
            200,
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": "Olá, tudo bem?"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        ),
        "stream_ok": sse_response(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-haiku-5-5",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 5, "output_tokens": 1},
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
                *(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                    for text in ("Olá, ", "tudo ", "bem?")
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 4},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        ),
        "rate_limited": json_response(
            429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        ),
        "timeout": raise_timeout,
    }

    def build(self, scenario: Scenario) -> LLMProvider:
        return AnthropicProvider(
            api_key="sk-ant-test",
            model="claude-haiku-5-5",
            timeout_seconds=5,
            max_retries=0,
            http_client=Recorder(self.responses[scenario]).client(),
        )


def _openai_chunk(content: str | None = None, finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }


class OpenAIHarness:
    # ClassVar: constante da classe, não um atributo de instância (evita estado compartilhado).
    responses: ClassVar[dict[Scenario, Handler]] = {
        "ok": json_response(
            200,
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Olá, tudo bem?",
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        ),
        "stream_ok": sse_response(
            [
                *((None, _openai_chunk(content=text)) for text in ("Olá, ", "tudo ", "bem?")),
                (None, _openai_chunk(finish_reason="stop")),
                (
                    None,
                    {
                        **_openai_chunk(),
                        "choices": [],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                    },
                ),
                (None, "[DONE]"),
            ]
        ),
        "rate_limited": json_response(
            429, {"error": {"message": "slow down", "type": "rate_limit_exceeded"}}
        ),
        "timeout": raise_timeout,
    }

    def build(self, scenario: Scenario) -> LLMProvider:
        return OpenAIProvider(
            api_key="sk-openai-test",
            model="gpt-test",
            timeout_seconds=5,
            max_retries=0,
            http_client=Recorder(self.responses[scenario]).client(),
        )


HARNESSES: dict[str, Callable[[], Harness]] = {
    "fake": FakeHarness,
    "anthropic": AnthropicHarness,
    "openai": OpenAIHarness,
}

ProviderBuilder = Callable[[Scenario], LLMProvider]


@pytest.fixture(params=sorted(HARNESSES))
async def build_provider(request: pytest.FixtureRequest) -> AsyncIterator[ProviderBuilder]:
    """Parametrizada: cada teste abaixo roda uma vez para cada provider."""
    harness = HARNESSES[request.param]()
    created: list[LLMProvider] = []

    def build(scenario: Scenario) -> LLMProvider:
        provider = harness.build(scenario)
        created.append(provider)
        return provider

    yield build
    for provider in created:
        await provider.aclose()


REQUEST = ChatRequest(
    messages=(Message(role=Role.USER, content="Diga olá."),),
    system="Responda em português.",
    max_output_tokens=100,
)


async def test_returns_a_valid_domain_response(build_provider: ProviderBuilder) -> None:
    provider = build_provider("ok")

    response = await provider.chat(REQUEST)

    assert isinstance(response, ChatResponse)
    assert response.content == "Olá, tudo bem?"
    assert response.provider == provider.name
    assert response.model
    assert response.stop_reason is StopReason.END_TURN
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


async def test_rate_limits_surface_as_retryable_domain_error(
    build_provider: ProviderBuilder,
) -> None:
    provider = build_provider("rate_limited")

    with pytest.raises(LLMRateLimitError) as exc_info:
        await provider.chat(REQUEST)

    assert exc_info.value.retryable is True


async def test_timeouts_surface_as_retryable_domain_error(build_provider: ProviderBuilder) -> None:
    provider = build_provider("timeout")

    with pytest.raises(LLMTimeoutError) as exc_info:
        await provider.chat(REQUEST)

    assert exc_info.value.retryable is True


async def test_never_leaks_sdk_exceptions(build_provider: ProviderBuilder) -> None:
    # Qualquer falha do provider deve chegar como erro NOSSO, nunca como exceção do SDK.
    for scenario in ("rate_limited", "timeout"):
        provider = build_provider(scenario)

        with pytest.raises(Exception) as exc_info:  # noqa: PT011 — o tipo é verificado abaixo
            await provider.chat(REQUEST)

        assert type(exc_info.value).__module__ == "app.core.errors"


async def test_can_be_closed(build_provider: ProviderBuilder) -> None:
    provider = build_provider("ok")

    await provider.aclose()  # não deve levantar exceção


# --- Streaming --------------------------------------------------------------------------------


async def test_stream_yields_deltas_then_exactly_one_completed(
    build_provider: ProviderBuilder,
) -> None:
    provider = build_provider("stream_ok")

    events = [event async for event in provider.stream(REQUEST)]

    *deltas, completed = events
    assert deltas, "o stream deve entregar o texto em pedaços"
    assert all(isinstance(event, TextDelta) for event in deltas)
    assert isinstance(completed, StreamCompleted)
    texts = "".join(event.text for event in deltas if isinstance(event, TextDelta))
    assert completed.response.content == texts == "Olá, tudo bem?"
    assert completed.response.provider == provider.name
    assert completed.response.stop_reason is StopReason.END_TURN


async def test_stream_failures_surface_as_domain_errors(build_provider: ProviderBuilder) -> None:
    provider = build_provider("rate_limited")

    with pytest.raises(LLMRateLimitError):
        _ = [event async for event in provider.stream(REQUEST)]


async def test_stream_can_be_abandoned_midway(build_provider: ProviderBuilder) -> None:
    provider = build_provider("stream_ok")
    stream = provider.stream(REQUEST)

    first = await anext(stream)
    await stream.aclose()  # consumidor desiste: não deve levantar nem vazar a conexão

    assert isinstance(first, TextDelta)
