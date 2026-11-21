"""Ports: interfaces que o núcleo precisa e que a infraestrutura implementa.

Usamos ``typing.Protocol`` (tipagem estrutural): qualquer classe com os mesmos métodos e
assinaturas *é* um ``LLMProvider`` para o mypy, sem precisar herdar desta classe. Os adapters em
``app.infra`` não importam nada daqui além dos modelos — o acoplamento fica mínimo.
"""

from collections.abc import AsyncGenerator
from typing import Protocol

from app.domain.llm import ChatRequest, ChatResponse, StreamEvent


class LLMProvider(Protocol):
    @property
    def name(self) -> str:
        """Identificador do provedor (ex.: ``"anthropic"``), usado em logs e respostas."""
        ...

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Envia a conversa e devolve a resposta completa (sem streaming)."""
        ...

    def stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        """Envia a conversa e devolve a resposta aos poucos, conforme é gerada.

        Contrato: zero ou mais ``TextDelta`` seguidos de exatamente um ``StreamCompleted``, cujo
        ``response.content`` é a concatenação dos deltas. Implementações são *async generators*
        (``async def`` com ``yield``). Devolver ``AsyncGenerator`` (e não só ``AsyncIterator``)
        garante ``aclose()``: quem consome pode encerrar o stream no meio e liberar a conexão.
        """
        ...

    async def aclose(self) -> None:
        """Libera recursos (conexões HTTP). Chamado no shutdown da aplicação."""
        ...
