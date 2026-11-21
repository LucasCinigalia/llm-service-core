"""Testes da configuração.

Usamos ``_env_file=None`` para que um ``.env`` local na sua máquina não interfira nos testes,
e ``monkeypatch`` para definir variáveis de ambiente que são desfeitas ao fim de cada teste.
"""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Environment, Settings, get_settings


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults_work_without_any_configuration() -> None:
    settings = make_settings()

    assert settings.environment is Environment.LOCAL
    assert settings.debug is False
    assert settings.llm.default_provider == "fake"


def test_reads_values_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = make_settings()

    assert settings.environment is Environment.STAGING
    assert settings.log_level == "DEBUG"


def test_nested_values_use_double_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM__ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM__REQUEST_TIMEOUT_SECONDS", "30")

    settings = make_settings()

    assert settings.llm.default_provider == "anthropic"
    assert settings.llm.request_timeout_seconds == 30.0


def test_api_keys_are_masked_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__OPENAI_API_KEY", "sk-super-secret")

    settings = make_settings()

    assert isinstance(settings.llm.openai_api_key, SecretStr)
    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in str(settings.model_dump())
    assert settings.llm.openai_api_key.get_secret_value() == "sk-super-secret"


def test_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError, match="log_level"):
        make_settings()


def test_fails_fast_when_default_provider_has_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM__DEFAULT_PROVIDER", "openai")

    with pytest.raises(ValidationError, match="LLM__OPENAI_API_KEY"):
        make_settings()


def test_debug_is_forbidden_in_production() -> None:
    with pytest.raises(ValidationError, match="debug must be disabled"):
        make_settings(environment="production", debug=True)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [("local", False), ("test", True), ("staging", True), ("production", True)],
)
def test_json_logs_everywhere_except_local(environment: str, expected: bool) -> None:
    assert make_settings(environment=environment).json_logs is expected


def test_get_settings_returns_cached_instance() -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()


def test_max_output_tokens_defaults_to_16000() -> None:
    assert make_settings().llm.max_output_tokens == 16000


def test_max_output_tokens_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__MAX_OUTPUT_TOKENS", "4096")
    assert make_settings().llm.max_output_tokens == 4096


@pytest.mark.parametrize("value", ["0", "-5", "32001"])
def test_max_output_tokens_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LLM__MAX_OUTPUT_TOKENS", value)
    with pytest.raises(ValidationError, match="max_output_tokens"):
        make_settings()


@pytest.mark.parametrize("value", ["1", "32000"])
def test_max_output_tokens_accepts_boundaries(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM__MAX_OUTPUT_TOKENS", value)
    assert make_settings().llm.max_output_tokens == int(value)


def test_anthropic_defaults_to_haiku_with_model_default_effort() -> None:
    llm = make_settings().llm

    assert llm.anthropic_model == "claude-haiku-5-5"
    assert llm.anthropic_effort is None
    assert llm.anthropic_refusal_fallback is False


def test_rejects_unknown_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__ANTHROPIC_EFFORT", "extreme")

    with pytest.raises(ValidationError, match="anthropic_effort"):
        make_settings()
