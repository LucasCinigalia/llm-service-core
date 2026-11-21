"""Ponto de entrada da aplicação (app factory).

Rodar localmente::

    uv run uvicorn app.main:create_app --factory --reload --no-access-log
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import chat, chat_stream, health
from app.core.config import Environment, Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.domain.ports import LLMProvider
from app.infra.llm.factory import build_llm_provider

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup (antes do ``yield``) e shutdown (depois). Aqui entram pools e clientes HTTP."""
    settings: Settings = app.state.settings
    logger.info(
        "app_started",
        version=__version__,
        environment=settings.environment,
        llm_provider=app.state.llm_provider.name,
    )
    yield
    await app.state.llm_provider.aclose()
    logger.info("app_stopped")


def create_app(
    settings: Settings | None = None,
    *,
    llm_provider: LLMProvider | None = None,
) -> FastAPI:
    """Monta a aplicação (composition root).

    ``llm_provider`` permite injetar um provider pronto — usado nos testes com o fake programado.
    Sem ele, o provider é escolhido pela configuração (``LLM__DEFAULT_PROVIDER``).
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs, service=settings.app_name)

    # Documentação interativa (/docs) desligada em produção: não expor a superfície da API.
    is_production = settings.environment is Environment.PRODUCTION
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None,
        openapi_url=None if is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.llm_provider = llm_provider or build_llm_provider(settings.llm)

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(chat_stream.router)

    return app
