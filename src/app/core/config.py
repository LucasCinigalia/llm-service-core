"""Configuração da aplicação, lida de variáveis de ambiente (12-Factor App).

Ordem de prioridade (maior vence):
    1. Argumentos passados diretamente para ``Settings(...)`` (útil em testes)
    2. Variáveis de ambiente
    3. Arquivo ``.env``
    4. Valores padrão definidos aqui
"""

from enum import StrEnum
from functools import lru_cache
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


LLMProviderName = Literal["fake", "anthropic", "openai"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
# Quanto o modelo raciocina antes de responder: menos effort = mais rápido e mais barato.
AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class LLMSettings(BaseModel):
    """Configurações dos provedores de LLM (variáveis com prefixo ``LLM__``)."""

    default_provider: LLMProviderName = "fake"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    # Padrão generoso: em modelos com raciocínio (thinking), os tokens de raciocínio também contam
    # no limite. Um valor baixo corta a resposta antes do texto final.
    max_output_tokens: int = Field(default=16000, ge=1, le=32000)
    # Retentativas feitas pelo próprio SDK (falhas transitórias). Resiliência completa: commit 15.
    max_retries: int = Field(default=2, ge=0, le=10)

    # Haiku: o modelo mais rápido e barato da Anthropic, ótimo para estudar e para volume.
    # Troque por "claude-sonnet-5-5" ou "claude-opus-5-5" quando precisar de mais capacidade.
    anthropic_model: str = "claude-haiku-5-5"
    # None = padrão do modelo (no Haiku 5.5 e no Opus 5.5, "medium").
    anthropic_effort: AnthropicEffort | None = None
    # Fallback de recusa no servidor: se o modelo recusar por política de segurança, a API refaz
    # a chamada em outro modelo. Ligue só em modelos que suportam (Opus, Sonnet, Fable) —
    # o Haiku 5.5 não tem fallback e recusa a requisição se receber uma lista de modelos.
    anthropic_refusal_fallback: bool = False

    # Modelos da OpenAI mudam rápido: confira a lista atual em https://platform.openai.com/docs/models
    openai_model: str = "gpt-6-luna"
    # Para APIs compatíveis com OpenAI: Azure OpenAI, vLLM, Ollama, LM Studio, Groq...
    openai_base_url: str | None = None
    # Só para modelos de raciocínio. None = padrão do modelo.
    openai_reasoning_effort: OpenAIReasoningEffort | None = None

    @model_validator(mode="after")
    def _require_key_for_default_provider(self) -> Self:
        required_keys: dict[str, SecretStr | None] = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }
        if self.default_provider in required_keys and required_keys[self.default_provider] is None:
            env_var = f"LLM__{self.default_provider.upper()}_API_KEY"
            msg = f"default_provider={self.default_provider!r} requires {env_var} to be set"
            raise ValueError(msg)
        if self.default_provider == "openai" and not self.openai_model:
            msg = "default_provider='openai' requires a non-empty LLM__OPENAI_MODEL"
            raise ValueError(msg)
        return self


class Settings(BaseSettings):
    """Configuração raiz. Cada grupo aninhado vira um prefixo: ``LLM__DEFAULT_PROVIDER``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "llm-service-core"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: LogLevel = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)

    @property
    def json_logs(self) -> bool:
        """Logs legíveis para humanos só no dev local; JSON em qualquer outro ambiente."""
        return self.environment is not Environment.LOCAL

    @model_validator(mode="after")
    def _forbid_debug_in_production(self) -> Self:
        if self.environment is Environment.PRODUCTION and self.debug:
            msg = "debug must be disabled in production"
            raise ValueError(msg)
        return self


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única de configuração (lida uma vez, reutilizada depois)."""
    return Settings()
