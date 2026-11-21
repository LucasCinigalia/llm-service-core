from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import REQUEST_ID_HEADER
from app.core.errors import LLMRateLimitError, LLMUnavailableError
from app.infra.llm.fake import FailMidStream, FakeLLMProvider
from stream_helpers import parse_sse

LogReader = Callable[[], list[dict[str, Any]]]  # tipo da fixture `logs` (conftest.py)


def body(**overrides: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "Olá!"}], **overrides}


def post_stream(client: TestClient, json: dict[str, Any]) -> tuple[int, dict[str, str], str]:
    with client.stream("POST", "/v1/chat/stream", json=json) as response:
        return response.status_code, dict(response.headers), response.read().decode()


def test_streams_deltas_then_done(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    fake_llm.enqueue("Olá, tudo bem?")

    status, headers, raw = post_stream(client, body())

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    assert REQUEST_ID_HEADER.lower() in headers
    events = parse_sse(raw)
    *deltas, (done_name, done) = events
    assert [name for name, _ in deltas] == ["delta", "delta", "delta"]
    assert "".join(data["text"] for _, data in deltas) == "Olá, tudo bem?"
    assert done_name == "done"
    assert done["content"] == "Olá, tudo bem?"
    assert done["stop_reason"] == "end_turn"
    assert set(done["usage"]) == {"input_tokens", "output_tokens", "total_tokens"}


def test_applies_the_same_defaults_as_chat(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    post_stream(client, body())

    [sent] = fake_llm.requests
    assert sent.max_output_tokens == 16000


def test_validation_errors_are_plain_http_errors(
    client: TestClient, fake_llm: FakeLLMProvider
) -> None:
    status, headers, _ = post_stream(client, {"messages": []})

    assert status == 422
    assert headers["content-type"] == "application/json"
    assert fake_llm.requests == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (LLMRateLimitError("slow down"), 503, "llm_rate_limited"),
        (LLMUnavailableError("down"), 503, "llm_unavailable"),
    ],
)
def test_failures_before_the_first_event_are_plain_http_errors(
    client: TestClient,
    fake_llm: FakeLLMProvider,
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    # Nada foi enviado ainda: dá para responder com o status certo e o envelope JSON.
    fake_llm.enqueue(error)

    status, headers, raw = post_stream(client, body())

    assert status == expected_status
    assert headers["content-type"] == "application/json"
    assert expected_code in raw


def test_failure_mid_stream_becomes_an_error_event(
    client: TestClient, fake_llm: FakeLLMProvider
) -> None:
    fake_llm.enqueue(FailMidStream(LLMUnavailableError("overloaded"), after_chunks=2))

    status, headers, raw = post_stream(client, body())

    # O 200 já tinha sido enviado: o erro chega como evento, depois do texto parcial.
    assert status == 200
    names = [name for name, _ in parse_sse(raw)]
    assert names == ["delta", "delta", "error"]
    _, error = parse_sse(raw)[-1]
    assert error["code"] == "llm_unavailable"
    assert error["request_id"] == headers[REQUEST_ID_HEADER.lower()]


def test_unexpected_exception_mid_stream_does_not_leak_details(
    client: TestClient, fake_llm: FakeLLMProvider, logs: LogReader
) -> None:
    fake_llm.enqueue(FailMidStream(RuntimeError("secret internal detail"), after_chunks=1))
    logs()

    status, _, raw = post_stream(client, body())

    assert status == 200
    _, error = parse_sse(raw)[-1]
    assert error["code"] == "internal_error"
    assert "secret internal detail" not in raw
    assert any(r["event"] == "unhandled_stream_exception" for r in logs())


def test_logs_ttft_and_latency(
    client: TestClient, fake_llm: FakeLLMProvider, logs: LogReader
) -> None:
    fake_llm.enqueue("um dois três")
    logs()

    post_stream(client, body())

    [completed] = [r for r in logs() if r["event"] == "llm_stream_completed"]
    assert completed["provider"] == "fake"
    assert 0 <= completed["ttft_ms"] <= completed["latency_ms"]
    assert completed["output_tokens"] > 0


def test_logs_mid_stream_failures(
    client: TestClient, fake_llm: FakeLLMProvider, logs: LogReader
) -> None:
    fake_llm.enqueue(FailMidStream(LLMUnavailableError("overloaded"), after_chunks=1))
    logs()

    post_stream(client, body())

    [failed] = [r for r in logs() if r["event"] == "llm_stream_failed"]
    assert failed["error_code"] == "llm_unavailable"
    assert failed["ttft_ms"] is not None  # o primeiro token chegou antes da falha
