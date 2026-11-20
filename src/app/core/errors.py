"""Hierarquia de erros da aplicação.

Os erros aqui são **independentes de HTTP**: dizem *o que* deu errado (``code``), não *como*
responder. A tradução para status HTTP fica na camada ``api`` (``app.api.errors``). Assim, os
mesmos erros servem para um worker, um CLI ou um servidor MCP — que não falam HTTP.
"""

from typing import Any


class AppError(Exception):
    """Erro base. Toda exceção esperada da aplicação deve herdar desta classe."""

    code: str = "internal_error"
    # Tentar de novo pode dar certo? (falha transitória). Usado pelo retry do commit 15.
    retryable: bool = False

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


# --- Erros de provedores de LLM -------------------------------------------------------------
# Os adapters traduzem as exceções de cada SDK para estas classes. O resto da aplicação nunca vê
# uma exceção da Anthropic ou da OpenAI.


class LLMProviderError(ExternalServiceError):
    """Falha genérica do provedor de LLM."""

    code = "llm_provider_error"


class LLMTimeoutError(LLMProviderError):
    code = "llm_timeout"
    retryable = True


class LLMRateLimitError(LLMProviderError):
    """O *nosso* acesso ao provedor foi limitado (não o cliente da nossa API)."""

    code = "llm_rate_limited"
    retryable = True


class LLMUnavailableError(LLMProviderError):
    """Provedor fora do ar, sobrecarregado ou inalcançável (5xx, 529, erro de conexão)."""

    code = "llm_unavailable"
    retryable = True


class LLMRequestRejectedError(InvalidInputError):
    """O provedor recusou a requisição como inválida (ex.: conversa maior que o contexto)."""

    code = "llm_request_rejected"


class LLMConfigurationError(AppError):
    """Erro nosso, não do cliente nem do provedor.

    Ex.: chave de API inválida, sem permissão, modelo inexistente.
    """

    code = "llm_configuration_error"
