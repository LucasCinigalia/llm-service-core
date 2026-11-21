"""Apoio para testar streams (async iterators de ``StreamEvent``)."""

import json
from collections.abc import AsyncIterator
from typing import Any

from app.domain.llm import StreamEvent, TextDelta


async def drain_texts(stream: AsyncIterator[StreamEvent], into: list[str]) -> None:
    """Consome o stream guardando os textos em ``into`` — inclusive se ele falhar no meio.

    Útil dentro de ``pytest.raises``: o bloco fica com um único comando e, depois da exceção,
    ``into`` mostra o que chegou ao consumidor antes da falha.
    """
    async for event in stream:
        if isinstance(event, TextDelta):
            into.append(event.text)


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Lê um corpo ``text/event-stream`` como um cliente faria: lista de ``(evento, dados)``."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        name, data = "message", ""
        for line in block.splitlines():
            field, _, value = line.partition(":")
            if field == "event":
                name = value.strip()
            elif field == "data":
                data += value.strip()
        events.append((name, json.loads(data)))
    return events
