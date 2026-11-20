"""Escolhe e constrói o adapter de LLM a partir da configuração.

É o único lugar que conhece *todos* os adapters. Adicionar um provider novo = criar o adapter e
registrar aqui; nenhum service ou rota muda.
"""

from app.core.config import LLMSettings
from app.domain.ports import LLMProvider
from app.infra.llm.fake import FakeLLMProvider


class ProviderNotAvailableError(RuntimeError):
    """Erro de configuração: o provider pedido ainda não tem adapter."""


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    match settings.default_provider:
        case "fake":
            return FakeLLMProvider()
        case "anthropic" | "openai":
            # Adapters chegam nos commits 10 e 11.
            msg = f"LLM provider {settings.default_provider!r} is not implemented yet"
            raise ProviderNotAvailableError(msg)
