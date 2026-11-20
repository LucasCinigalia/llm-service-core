"""Fixtures da camada HTTP.

Criamos a aplicação com ``Settings`` explícitas (sem ``.env``), e o ``TestClient`` dentro de um
``with`` para que o *lifespan* (startup/shutdown) rode como em produção.
"""

import io
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

LogReader = Callable[[], list[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, environment="test")


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def logs(app: FastAPI) -> LogReader:
    """Redireciona os logs da aplicação para um buffer em memória.

    Retorna uma função que devolve os logs JSON emitidos desde a última chamada.
    """
    buffer = io.StringIO()
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(buffer)

    def read() -> list[dict[str, Any]]:
        lines = buffer.getvalue().splitlines()
        buffer.seek(0)
        buffer.truncate()
        return [json.loads(line) for line in lines if line]

    return read
