import pytest

from app.core.config import LLMSettings
from app.core.errors import ExternalServiceError
from app.domain.llm import ChatRequest, Message, Role
from app.infra.llm.fake import FakeLLMProvider
from app.services.chat import ChatService


def make_request(**kwargs: object) -> ChatRequest:
    return ChatRequest(messages=(Message(role=Role.USER, content="oi"),), **kwargs)


@pytest.fixture
def fake() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def service(fake: FakeLLMProvider) -> ChatService:
    return ChatService(fake, LLMSettings(max_output_tokens=256))


async def test_returns_the_provider_response(service: ChatService) -> None:
    response = await service.chat(make_request())

    assert response.content == "[fake] oi"


async def test_applies_default_max_output_tokens_from_settings(
    service: ChatService, fake: FakeLLMProvider
) -> None:
    await service.chat(make_request())

    [sent] = fake.requests
    assert sent.max_output_tokens == 256


async def test_keeps_max_output_tokens_chosen_by_the_caller(
    service: ChatService, fake: FakeLLMProvider
) -> None:
    await service.chat(make_request(max_output_tokens=50))

    [sent] = fake.requests
    assert sent.max_output_tokens == 50


async def test_does_not_mutate_the_callers_request(
    service: ChatService, fake: FakeLLMProvider
) -> None:
    original = make_request()

    await service.chat(original)

    assert original.max_output_tokens is None
    assert fake.requests[0] is not original


async def test_propagates_provider_errors(service: ChatService, fake: FakeLLMProvider) -> None:
    fake.enqueue(ExternalServiceError("provider down"))

    with pytest.raises(ExternalServiceError, match="provider down"):
        await service.chat(make_request())
