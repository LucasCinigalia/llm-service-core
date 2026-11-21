"""Schemas HTTP (DTOs) do chat: o contrato público da API.

São separados dos modelos de domínio de propósito:
- a API pode evoluir (``/v2``) sem mudar o domínio, e vice-versa;
- aqui ficam limites de *borda* (tamanho máximo, contra abuso) e exemplos para o ``/docs``;
- o domínio usa ``tuple`` e objetos imutáveis; a API recebe JSON comum.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.llm import ChatRequest, ChatResponse, Message, Role, StopReason


class MessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequestBody(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Explique RAG em uma frase."}],
                    "system": "Você é um assistente técnico e conciso.",
                    "temperature": 0.2,
                }
            ]
        },
    )

    messages: list[MessageIn] = Field(min_length=1, max_length=200)
    system: str | None = Field(default=None, max_length=50_000)
    model: str | None = Field(default=None, max_length=200)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    def to_domain(self) -> ChatRequest:
        return ChatRequest(
            messages=tuple(Message(role=Role(m.role), content=m.content) for m in self.messages),
            system=self.system,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponseBody(BaseModel):
    content: str
    model: str
    provider: str
    stop_reason: StopReason
    usage: UsageOut

    @classmethod
    def from_domain(cls, response: ChatResponse) -> "ChatResponseBody":
        return cls(
            content=response.content,
            model=response.model,
            provider=response.provider,
            stop_reason=response.stop_reason,
            usage=UsageOut(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )


class StreamDeltaOut(BaseModel):
    """Dados do evento SSE ``delta``: um pedaço do texto, na ordem."""

    text: str
