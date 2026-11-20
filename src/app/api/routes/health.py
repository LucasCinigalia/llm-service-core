"""Health check: usado por load balancers e orquestradores (ECS, Kubernetes, Docker)."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.api.dependencies import SettingsDep
from app.core.config import Environment

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: Environment


@router.get("/health")
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness: o processo está de pé e respondendo."""
    return HealthResponse(version=__version__, environment=settings.environment)
