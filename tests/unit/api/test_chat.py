from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import REQUEST_ID_HEADER
from app.core.errors import ExternalServiceError
from app.domain.llm import Role
from app.infra.llm.fake import FakeLLMProvider

LogReader = Callable[[], list[dict[str, Any]]]  # tipo da fixture `logs` (conftest.py)


def body(**overrides: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "Olá!"}], **overrides}


def test_returns_completion_from_the_configured_provider(client: TestClient) -> None:
    response = client.post("/v1/chat", json=body())

    assert response.status_code == 200
    assert response.json() == {
        "content": "[fake] Olá!",
        "model": "fake-model",
        "provider": "fake",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }


def test_translates_http_body_into_domain_request(
    client: TestClient, fake_llm: FakeLLMProvider
) -> None:
    client.post(
        "/v1/chat",
        json={
            "messages": [
                {"role": "user", "content": "Oi"},
                {"role": "assistant", "content": "Olá! Como posso ajudar?"},
                {"role": "user", "content": "Resuma isto."},
            ],
            "system": "Seja conciso.",
            "model": "fake-large",
            "temperature": 0.3,
        },
    )

    [sent] = fake_llm.requests
    assert [m.role for m in sent.messages] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert sent.system == "Seja conciso."
    assert sent.model == "fake-large"
    assert sent.temperature == 0.3
    assert sent.max_output_tokens == 1024  # padrão da configuração aplicado pelo service


@pytest.mark.parametrize(
    ("invalid_body", "field"),
    [
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "system", "content": "x"}]}, "role"),
        ({"messages": [{"role": "user", "content": ""}]}, "content"),
        (body(temperature=5), "temperature"),
        (body(max_output_tokens=0), "max_output_tokens"),
        (body(unknown_field=True), "unknown_field"),
    ],
)
def test_rejects_invalid_bodies_with_standard_envelope(
    client: TestClient, fake_llm: FakeLLMProvider, invalid_body: dict[str, Any], field: str
) -> None:
    response = client.post("/v1/chat", json=invalid_body)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert field in str(error["details"])
    assert fake_llm.requests == []  # nada chegou ao provider


def test_provider_failure_returns_502(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    fake_llm.enqueue(ExternalServiceError("Anthropic is overloaded"))

    response = client.post("/v1/chat", json=body())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "external_service_error"


def test_logs_llm_metrics_with_request_id(client: TestClient, logs: LogReader) -> None:
    logs()

    response = client.post("/v1/chat", json=body())

    [completed] = [r for r in logs() if r["event"] == "llm_request_completed"]
    assert completed["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert completed["provider"] == "fake"
    assert completed["model"] == "fake-model"
    assert completed["input_tokens"] == 1
    assert completed["output_tokens"] == 2
    assert completed["latency_ms"] >= 0
    assert "Olá" not in str(completed)  # conteúdo da conversa não vai para os logs


def test_warns_when_response_is_truncated(
    client: TestClient, fake_llm: FakeLLMProvider, logs: LogReader
) -> None:
    fake_llm.enqueue("x" * 400)
    logs()

    response = client.post("/v1/chat", json=body(max_output_tokens=10))

    assert response.json()["stop_reason"] == "max_tokens"
    [warning] = [r for r in logs() if r["event"] == "llm_response_truncated"]
    assert warning["level"] == "warning"
    assert warning["max_output_tokens"] == 10


def test_logs_provider_failures(
    client: TestClient, fake_llm: FakeLLMProvider, logs: LogReader
) -> None:
    fake_llm.enqueue(ExternalServiceError("down"))
    logs()

    client.post("/v1/chat", json=body())

    [failed] = [r for r in logs() if r["event"] == "llm_request_failed"]
    assert failed["error_code"] == "external_service_error"
    assert failed["provider"] == "fake"
