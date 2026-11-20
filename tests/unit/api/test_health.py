from fastapi.testclient import TestClient

from app import __version__
from app.core.config import Settings
from app.main import create_app


def test_health_reports_status_version_and_environment(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "environment": "test"}


def test_docs_are_available_outside_production(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_are_disabled_in_production() -> None:
    settings = Settings(_env_file=None, environment="production")

    with TestClient(create_app(settings)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
