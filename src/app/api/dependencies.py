"""Dependências injetáveis nas rotas via ``Depends``."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings


def get_app_settings(request: Request) -> Settings:
    """Settings da instância da aplicação (definidas em ``create_app``)."""
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
