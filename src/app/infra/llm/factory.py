"""Escolhe e constrói o adapter de LLM a partir da configuração.

É o único lugar que conhece *todos* os adapters. Adicionar um provider novo = criar o adapter e
registrar aqui; nenhum service ou rota muda.
"""

from app.core.config import LLMSettings
from app.domain.ports import LLMProvider
from app.infra.llm.anthropic_provider import AnthropicProvider
from app.infra.llm.fake import FakeLLMProvider
from app.infra.llm.openai_provider import OpenAIProvider


class ProviderNotAvailableError(RuntimeError):
    """Erro de configuração: o provider pedido não pode ser construído."""


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
                effort=settings.anthropic_effort,
                timeout_seconds=settings.request_timeout_seconds,
                max_retries=settings.max_retries,
                refusal_fallback=settings.anthropic_refusal_fallback,
            )
        case "openai":
            # Já validados em LLMSettings; a checagem aqui é para o mypy (tipos opcionais).
            if settings.openai_api_key is None or not settings.openai_model:
                msg = "LLM__OPENAI_API_KEY and LLM__OPENAI_MODEL are required for openai"
                raise ProviderNotAvailableError(msg)
            return OpenAIProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                reasoning_effort=settings.openai_reasoning_effort,
                timeout_seconds=settings.request_timeout_seconds,
                max_retries=settings.max_retries,
            )
