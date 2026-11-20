"""Logging estruturado com structlog.

- Em desenvolvimento: saída colorida e legível para humanos.
- Em produção: uma linha JSON por evento, pronta para Datadog/CloudWatch/Loki/ELK.

Logs da biblioteca padrão (``logging``) — usados por uvicorn, httpx, SDKs etc. — passam pelo
mesmo pipeline de processadores, então *todos* os logs do processo saem no mesmo formato.

Uso::

    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("llm_request_completed", provider="anthropic", latency_ms=812)
"""

import logging
import sys

import structlog
from structlog.typing import Processor


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configura structlog e o ``logging`` da biblioteca padrão. Chamar uma vez no startup."""
    shared_processors: list[Processor] = [
        # Injeta o contexto da request atual (ex.: request_id) em todos os logs.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # Para logs da stdlib: preserva campos passados via `extra={...}`.
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Processor
    if json_logs:
        # Exceções viram uma estrutura (tipo, mensagem, frames) em vez de texto multilinha.
        shared_processors.append(structlog.processors.dict_tracebacks)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            # Entrega o evento ao ProcessorFormatter abaixo, que faz a renderização final.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Processadores aplicados somente a logs que NÃO vieram do structlog (stdlib).
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retorna um logger estruturado. Convenção: ``get_logger(__name__)``."""
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger
