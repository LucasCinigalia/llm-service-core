from http import HTTPStatus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import error_response, status_for
from app.api.middleware import REQUEST_ID_HEADER
from app.core.errors import (
    AppError,
    ExternalServiceError,
    InvalidInputError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUnavailableError,
    NotFoundError,
)


class ConversationNotFoundError(NotFoundError):
    """Exemplo de erro específico de um projeto, herdando de um erro genérico."""


@pytest.fixture
def client_with_routes(app: FastAPI) -> TestClient:
    @app.get("/items/{item_id}")
    async def get_item(item_id: int) -> dict[str, int]:
        if item_id == 0:
            raise NotFoundError("Item not found.", details={"item_id": item_id})
        return {"item_id": item_id}

    return TestClient(app)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (NotFoundError("x"), 404),
        (ConversationNotFoundError("x"), 404),  # herda o status da classe-mãe
        (InvalidInputError("x"), 422),
        (ExternalServiceError("x"), 502),
        (AppError("x"), 500),
        (LLMTimeoutError("x"), 504),
        (LLMRateLimitError("x"), 503),
        (LLMUnavailableError("x"), 503),
        (LLMRequestRejectedError("x"), 422),  # herda de InvalidInputError
        (LLMConfigurationError("x"), 500),  # erro nosso: herda de AppError
    ],
)
def test_maps_domain_errors_to_http_status(error: AppError, expected_status: int) -> None:
    assert status_for(error) == expected_status


def test_app_error_uses_standard_envelope(client_with_routes: TestClient) -> None:
    response = client_with_routes.get("/items/0")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Item not found.",
            "request_id": response.headers[REQUEST_ID_HEADER],
            "details": {"item_id": 0},
        }
    }


def test_validation_error_uses_standard_envelope(client_with_routes: TestClient) -> None:
    response = client_with_routes.get("/items/not-a-number")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["errors"][0]["loc"] == ["path", "item_id"]


def test_validation_error_does_not_echo_the_rejected_input(client_with_routes: TestClient) -> None:
    response = client_with_routes.get("/items/my-secret-token")

    [error] = response.json()["error"]["details"]["errors"]
    assert "input" not in error
    assert "my-secret-token" not in response.text


def test_unknown_route_uses_standard_envelope(client_with_routes: TestClient) -> None:
    response = client_with_routes.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_error_response_status_is_a_plain_int() -> None:
    # Um IntEnum "vazaria" para o log de acesso como <HTTPStatus.NOT_FOUND: 404>.
    response = error_response(HTTPStatus.NOT_FOUND, "not_found", "x")

    assert type(response.status_code) is int
