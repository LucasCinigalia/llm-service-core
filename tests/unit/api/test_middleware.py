import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import REQUEST_ID_HEADER, resolve_request_id

LogReader = Callable[[], list[dict[str, Any]]]  # tipo da fixture `logs` (conftest.py)


def test_generates_request_id_when_client_does_not_send_one(client: TestClient) -> None:
    response = client.get("/health")

    request_id = response.headers[REQUEST_ID_HEADER]
    assert len(request_id) == 32
    assert int(request_id, 16) >= 0  # hexadecimal (uuid4().hex)


def test_echoes_valid_request_id_from_client(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "client-abc.123"})

    assert response.headers[REQUEST_ID_HEADER] == "client-abc.123"


@pytest.mark.parametrize(
    "unsafe_id",
    ["", "has spaces", "line\nbreak", "x" * 129, '{"json": true}'],
)
def test_replaces_unsafe_request_ids(unsafe_id: str) -> None:
    assert resolve_request_id(unsafe_id) != unsafe_id


def test_logs_every_request_with_request_id_and_duration(
    client: TestClient, logs: LogReader
) -> None:
    logs()  # descarta os logs do startup

    response = client.get("/health?token=secret")

    [access_log] = [r for r in logs() if r["event"] == "http_request_completed"]
    assert access_log["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert access_log["method"] == "GET"
    assert access_log["path"] == "/health"
    assert access_log["status_code"] == 200
    assert access_log["duration_ms"] >= 0
    assert "secret" not in json.dumps(access_log)


def test_request_context_does_not_leak_between_requests(
    client: TestClient, logs: LogReader
) -> None:
    logs()

    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    access_logs = [r for r in logs() if r["event"] == "http_request_completed"]
    assert first != second
    assert [r["request_id"] for r in access_logs] == [first, second]


def test_unhandled_exception_returns_generic_500(app: FastAPI, logs: LogReader) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("database password is hunter2")

    with TestClient(app) as client:
        logs()
        response = client.get("/boom")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    assert "hunter2" not in response.text  # detalhes internos não vazam para o cliente
    assert error["request_id"] == response.headers[REQUEST_ID_HEADER]

    records = logs()
    [exception_log] = [r for r in records if r["event"] == "unhandled_exception"]
    assert exception_log["exception"][0]["exc_type"] == "RuntimeError"
    [access_log] = [r for r in records if r["event"] == "http_request_completed"]
    assert access_log["status_code"] == 500
