"""Dependências injetáveis nas rotas via ``Depends``.

As instâncias "caras" (settings, provider) são criadas uma vez em ``create_app`` e guardadas em
``app.state``; aqui só as buscamos. O ``ChatService`` é leve e criado por request.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.domain.ports import LLMProvider
from app.services.chat import ChatService


def get_app_settings(request: Request) -> Settings:
    """Settings da instância da aplicação (definidas em ``create_app``)."""
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def get_llm_provider(request: Request) -> LLMProvider:
    provider: LLMProvider = request.app.state.llm_provider
    return provider


LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]


def get_chat_service(provider: LLMProviderDep, settings: SettingsDep) -> ChatService:
    return ChatService(provider, settings.llm)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
