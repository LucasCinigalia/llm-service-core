import asyncio
import time

import pytest

from app.core.errors import ExternalServiceError
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
from app.domain.ports import LLMProvider
from app.infra.llm.fake import FailMidStream, FakeLLMProvider, estimate_tokens
from stream_helpers import drain_texts


def make_request(text: str = "olá", **kwargs: object) -> ChatRequest:
    return ChatRequest(
        messages=(Message(role=Role.USER, content=text),),
        **kwargs,
    )


def test_satisfies_the_llm_provider_port() -> None:
    provider: LLMProvider = FakeLLMProvider()  # checado pelo mypy

    assert provider.name == "fake"


async def test_echoes_last_message_when_nothing_is_scripted() -> None:
    provider = FakeLLMProvider()

    response = await provider.chat(make_request("qual a capital da França?"))

    assert response.content == "[fake] qual a capital da França?"
    assert response.provider == "fake"
    assert response.model == "fake-model"
    assert response.stop_reason is StopReason.END_TURN


async def test_returns_scripted_responses_in_order_then_falls_back_to_echo() -> None:
    provider = FakeLLMProvider(["primeira", "segunda"])

    contents = [(await provider.chat(make_request("x"))).content for _ in range(3)]

    assert contents == ["primeira", "segunda", "[fake] x"]


async def test_enqueue_adds_more_scripted_responses() -> None:
    provider = FakeLLMProvider(["a"])
    provider.enqueue("b", "c")

    contents = [(await provider.chat(make_request())).content for _ in range(3)]

    assert contents == ["a", "b", "c"]


async def test_raises_scripted_errors() -> None:
    provider = FakeLLMProvider([ExternalServiceError("provider down"), "recovered"])

    with pytest.raises(ExternalServiceError, match="provider down"):
        await provider.chat(make_request())
    assert (await provider.chat(make_request())).content == "recovered"


async def test_returns_full_scripted_response_unchanged() -> None:
    scripted = ChatResponse(
        content="cortado",
        model="custom",
        provider="fake",
        stop_reason=StopReason.CONTENT_FILTER,
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    provider = FakeLLMProvider([scripted])

    assert await provider.chat(make_request()) is scripted


async def test_records_every_request_for_later_inspection() -> None:
    provider = FakeLLMProvider()
    first, second = make_request("um"), make_request("dois", temperature=0.2)

    await provider.chat(first)
    await provider.chat(second)

    assert provider.requests == [first, second]


async def test_uses_requested_model_when_provided() -> None:
    response = await FakeLLMProvider().chat(make_request(model="fake-large"))

    assert response.model == "fake-large"


async def test_reports_estimated_token_usage() -> None:
    request = ChatRequest(
        system="Você é útil.",
        messages=(Message(role=Role.USER, content="a" * 40),),
    )

    response = await FakeLLMProvider(["b" * 20]).chat(request)

    assert response.usage.input_tokens == estimate_tokens(f"Você é útil. {'a' * 40}")
    assert response.usage.output_tokens == 5


async def test_simulates_truncation_by_max_output_tokens() -> None:
    provider = FakeLLMProvider(["x" * 100])  # ~25 tokens

    response = await provider.chat(make_request(max_output_tokens=5))

    assert response.stop_reason is StopReason.MAX_TOKENS
    assert len(response.content) == 20
    assert response.usage.output_tokens == 5


async def test_simulated_latency_does_not_block_the_event_loop() -> None:
    provider = FakeLLMProvider(latency_seconds=0.1)

    start = time.perf_counter()
    await asyncio.gather(*(provider.chat(make_request()) for _ in range(10)))
    elapsed = time.perf_counter() - start

    # 10 chamadas concorrentes de 0,1 s terminam em ~0,1 s, não em 1 s.
    assert elapsed < 0.5


@pytest.mark.parametrize(("text", "tokens"), [("", 1), ("abc", 1), ("abcd" * 10, 10)])
def test_estimate_tokens(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


# --- Streaming --------------------------------------------------------------------------------


async def collect(provider: FakeLLMProvider, request: ChatRequest) -> list[StreamEvent]:
    return [event async for event in provider.stream(request)]


async def test_stream_yields_word_chunks_then_completed() -> None:
    provider = FakeLLMProvider(["Olá, tudo bem?"])

    events = await collect(provider, make_request())

    *deltas, completed = events
    assert [d.text for d in deltas if isinstance(d, TextDelta)] == ["Olá, ", "tudo ", "bem?"]
    assert isinstance(completed, StreamCompleted)
    assert completed.response.content == "Olá, tudo bem?"


async def test_stream_raises_scripted_errors_before_any_event() -> None:
    provider = FakeLLMProvider([ExternalServiceError("down")])

    with pytest.raises(ExternalServiceError):
        await collect(provider, make_request())


async def test_stream_can_fail_midway() -> None:
    provider = FakeLLMProvider(
        [FailMidStream(ExternalServiceError("lost connection"), after_chunks=2, content="a b c d")]
    )
    received: list[str] = []

    with pytest.raises(ExternalServiceError, match="lost connection"):
        await drain_texts(provider.stream(make_request()), into=received)

    assert received == ["a ", "b "]


async def test_fail_mid_stream_raises_immediately_in_chat() -> None:
    provider = FakeLLMProvider([FailMidStream(ExternalServiceError("x"))])

    with pytest.raises(ExternalServiceError):
        await provider.chat(make_request())


async def test_counts_streams_aborted_by_the_consumer() -> None:
    provider = FakeLLMProvider(["uma resposta bem longa"])

    stream = provider.stream(make_request())
    await anext(stream)
    await stream.aclose()  # o consumidor desiste (ex.: cliente HTTP desconectou)

    assert provider.aborted_streams == 1


async def test_completed_streams_are_not_counted_as_aborted() -> None:
    provider = FakeLLMProvider()

    await collect(provider, make_request())

    assert provider.aborted_streams == 0
