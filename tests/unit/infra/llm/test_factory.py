import pytest
from pydantic import SecretStr

from app.core.config import LLMSettings
from app.infra.llm.anthropic_provider import AnthropicProvider
from app.infra.llm.factory import ProviderNotAvailableError, build_llm_provider
from app.infra.llm.fake import FakeLLMProvider


def test_builds_fake_provider_by_default() -> None:
    assert isinstance(build_llm_provider(LLMSettings()), FakeLLMProvider)


async def test_builds_anthropic_provider_when_configured() -> None:
    settings = LLMSettings(default_provider="anthropic", anthropic_api_key=SecretStr("sk-ant"))

    provider = build_llm_provider(settings)

    assert isinstance(provider, AnthropicProvider)
    await provider.aclose()


def test_fails_fast_for_providers_without_adapter() -> None:
    settings = LLMSettings(default_provider="openai", openai_api_key=SecretStr("sk"))

    with pytest.raises(ProviderNotAvailableError, match="openai"):
        build_llm_provider(settings)
