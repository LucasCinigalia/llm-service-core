"""Apoio para testar adapters de LLM sem rede.

Os SDKs (anthropic, openai) aceitam um ``httpx2.AsyncClient`` próprio. Passamos um cliente com
``MockTransport``: o SDK monta a requisição HTTP real, mas ela é entregue a uma função nossa em
vez de ir para a internet. Testamos o SDK de verdade — só a rede é simulada.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx2

Handler = Callable[[httpx2.Request], httpx2.Response]


class Recorder:
    """Transport falso que delega ao ``handler`` e guarda as requisições recebidas."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last_body(self) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(self.requests[-1].content)
        return body

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))


def json_response(status: int, body: dict[str, Any], **headers: str) -> Handler:
    return lambda _request: httpx2.Response(status, json=body, headers=headers)


def raise_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("timed out", request=request)


def raise_connection_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)
