import pytest
from pydantic import SecretStr

from app.core.config import LLMSettings
from app.infra.llm.factory import ProviderNotAvailableError, build_llm_provider
from app.infra.llm.fake import FakeLLMProvider


def test_builds_fake_provider_by_default() -> None:
    assert isinstance(build_llm_provider(LLMSettings()), FakeLLMProvider)


def test_fails_fast_for_providers_without_adapter() -> None:
    settings = LLMSettings(default_provider="anthropic", anthropic_api_key=SecretStr("sk"))

    with pytest.raises(ProviderNotAvailableError, match="anthropic"):
        build_llm_provider(settings)
