"""Ports: interfaces que o núcleo precisa e que a infraestrutura implementa.

Usamos ``typing.Protocol`` (tipagem estrutural): qualquer classe com os mesmos métodos e
assinaturas *é* um ``LLMProvider`` para o mypy, sem precisar herdar desta classe. Os adapters em
``app.infra`` não importam nada daqui além dos modelos — o acoplamento fica mínimo.
"""

from typing import Protocol

from app.domain.llm import ChatRequest, ChatResponse


class LLMProvider(Protocol):
    @property
    def name(self) -> str:
        """Identificador do provedor (ex.: ``"anthropic"``), usado em logs e respostas."""
        ...

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Envia a conversa e devolve a resposta completa (sem streaming)."""
        ...

    async def aclose(self) -> None:
        """Libera recursos (conexões HTTP). Chamado no shutdown da aplicação."""
        ...
