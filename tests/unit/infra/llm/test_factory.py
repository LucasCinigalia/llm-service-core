import pytest
from pydantic import SecretStr

from app.core.config import LLMSettings
from app.infra.llm.anthropic_provider import AnthropicProvider
from app.infra.llm.factory import ProviderNotAvailableError, build_llm_provider
from app.infra.llm.fake import FakeLLMProvider
from app.infra.llm.openai_provider import OpenAIProvider


def test_builds_fake_provider_by_default() -> None:
    assert isinstance(build_llm_provider(LLMSettings()), FakeLLMProvider)


async def test_builds_anthropic_provider_when_configured() -> None:
    settings = LLMSettings(default_provider="anthropic", anthropic_api_key=SecretStr("sk-ant"))

    provider = build_llm_provider(settings)

    assert isinstance(provider, AnthropicProvider)
    await provider.aclose()


async def test_builds_openai_provider_when_configured() -> None:
    settings = LLMSettings(
        default_provider="openai", openai_api_key=SecretStr("sk"), openai_model="gpt-test"
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, OpenAIProvider)
    await provider.aclose()


def test_factory_double_checks_required_openai_settings() -> None:
    # model_construct pula a validação: simula settings montadas sem passar pelo validator.
    settings = LLMSettings.model_construct(default_provider="openai", openai_api_key=None)

    with pytest.raises(ProviderNotAvailableError, match="LLM__OPENAI_MODEL"):
        build_llm_provider(settings)
