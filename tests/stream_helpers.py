"""Apoio para testar streams (async iterators de ``StreamEvent``)."""

from collections.abc import AsyncIterator

from app.domain.llm import StreamEvent, TextDelta


async def drain_texts(stream: AsyncIterator[StreamEvent], into: list[str]) -> None:
    """Consome o stream guardando os textos em ``into`` — inclusive se ele falhar no meio.

    Útil dentro de ``pytest.raises``: o bloco fica com um único comando e, depois da exceção,
    ``into`` mostra o que chegou ao consumidor antes da falha.
    """
    async for event in stream:
        if isinstance(event, TextDelta):
            into.append(event.text)
