from fastapi.testclient import TestClient

from app.core.config import Settings
from app.infra.llm.fake import FakeLLMProvider
from app.main import create_app


class ClosingSpy(FakeLLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_shutdown_closes_the_llm_provider(settings: Settings) -> None:
    provider = ClosingSpy()

    with TestClient(create_app(settings, llm_provider=provider)):
        assert provider.closed is False

    assert provider.closed is True
