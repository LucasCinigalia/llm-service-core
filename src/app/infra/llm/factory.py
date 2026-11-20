"""Escolhe e constrói o adapter de LLM a partir da configuração.

É o único lugar que conhece *todos* os adapters. Adicionar um provider novo = criar o adapter e
registrar aqui; nenhum service ou rota muda.
"""

from app.core.config import LLMSettings
from app.domain.ports import LLMProvider
from app.infra.llm.anthropic_provider import AnthropicProvider
from app.infra.llm.fake import FakeLLMProvider


class ProviderNotAvailableError(RuntimeError):
    """Erro de configuração: o provider pedido ainda não tem adapter."""


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    match settings.default_provider:
        case "fake":
            return FakeLLMProvider()
        case "anthropic":
            if settings.anthropic_api_key is None:  # já validado em LLMSettings; checagem p/ mypy
                msg = "LLM__ANTHROPIC_API_KEY is required for the anthropic provider"
                raise ProviderNotAvailableError(msg)
            return AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.anthropic_model,
                timeout_seconds=settings.request_timeout_seconds,
                max_retries=settings.max_retries,
                refusal_fallback=settings.anthropic_refusal_fallback,
            )
        case "openai":
            # Adapter chega no commit 11.
            msg = f"LLM provider {settings.default_provider!r} is not implemented yet"
            raise ProviderNotAvailableError(msg)
