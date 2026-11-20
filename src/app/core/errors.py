"""Hierarquia de erros da aplicação.

Os erros aqui são **independentes de HTTP**: dizem *o que* deu errado (``code``), não *como*
responder. A tradução para status HTTP fica na camada ``api`` (``app.api.errors``). Assim, os
mesmos erros servem para um worker, um CLI ou um servidor MCP — que não falam HTTP.
"""

from typing import Any


class AppError(Exception):
    """Erro base. Toda exceção esperada da aplicação deve herdar desta classe."""

    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    code = "not_found"


class InvalidInputError(AppError):
    code = "invalid_input"


class ExternalServiceError(AppError):
    """Falha em uma dependência externa (ex.: provedor de LLM indisponível)."""

    code = "external_service_error"
