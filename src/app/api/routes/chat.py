"""Rotas de chat. A rota é fina: traduz HTTP ↔ domínio e delega ao service."""

from fastapi import APIRouter

from app.api.dependencies import ChatServiceDep
from app.api.errors import ErrorResponse
from app.api.schemas.chat import ChatRequestBody, ChatResponseBody

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    responses={
        422: {"model": ErrorResponse, "description": "Requisição inválida"},
        502: {"model": ErrorResponse, "description": "Falha no provedor de LLM"},
    },
)
async def chat(body: ChatRequestBody, service: ChatServiceDep) -> ChatResponseBody:
    """Envia uma conversa ao LLM configurado e devolve a resposta completa."""
    response = await service.chat(body.to_domain())
    return ChatResponseBody.from_domain(response)
