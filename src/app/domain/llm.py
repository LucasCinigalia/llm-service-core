"""Modelos de domínio para conversar com LLMs, independentes de provedor.

São a "língua comum" da aplicação: cada adapter (Anthropic, OpenAI, fake) traduz estes modelos
para o formato do seu SDK e traduz a resposta de volta. O resto do código nunca vê tipos de SDK.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DomainModel(BaseModel):
    """Base dos modelos de domínio: imutáveis e estritos."""

    # frozen: objetos de valor não mudam depois de criados (seguro para compartilhar e cachear).
    # extra="forbid": campo desconhecido é erro — pega typos como `max_tokem=10`.
    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(DomainModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(DomainModel):
    """O que a aplicação quer do LLM. Campos ``None`` usam o padrão do provider/configuração."""

    messages: tuple[Message, ...] = Field(min_length=1)
    # O system prompt é separado das mensagens: a Anthropic o recebe num campo próprio e a OpenAI
    # como primeira mensagem — essa diferença é problema do adapter, não do domínio.
    system: str | None = None
    model: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class StopReason(StrEnum):
    """Por que o modelo parou de gerar, normalizado entre provedores."""

    END_TURN = "end_turn"  # terminou naturalmente
    MAX_TOKENS = "max_tokens"  # cortado pelo limite: a resposta pode estar incompleta
    STOP_SEQUENCE = "stop_sequence"  # encontrou uma sequência de parada
    CONTENT_FILTER = "content_filter"  # bloqueado por política de segurança do provedor


class Usage(DomainModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ChatResponse(DomainModel):
    content: str
    # O modelo que *de fato* respondeu (pode diferir do pedido: alias, fallback).
    model: str
    provider: str
    stop_reason: StopReason
    usage: Usage


# --- Streaming ----------------------------------------------------------------------------------
# Um stream é uma sequência de eventos: vários TextDelta (pedaços do texto, na ordem) e, por
# último, exatamente um StreamCompleted com a resposta consolidada (texto completo, tokens, etc.).


class TextDelta(DomainModel):
    type: Literal["text_delta"] = "text_delta"
    text: str = Field(min_length=1)


class StreamCompleted(DomainModel):
    type: Literal["completed"] = "completed"
    response: ChatResponse


StreamEvent = TextDelta | StreamCompleted
