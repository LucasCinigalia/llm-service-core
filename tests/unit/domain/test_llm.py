import pytest
from pydantic import ValidationError

from app.domain.llm import ChatRequest, ChatResponse, Message, Role, StopReason, Usage


def user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def test_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError, match="content"):
        Message(role=Role.USER, content="")


def test_message_accepts_role_as_string() -> None:
    # Útil ao converter JSON da API: "user" vira Role.USER.
    assert Message(role="assistant", content="hi").role is Role.ASSISTANT


def test_domain_models_are_immutable() -> None:
    message = user("hello")

    with pytest.raises(ValidationError, match="frozen"):
        message.content = "changed"  # type: ignore[misc]


def test_domain_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="max_tokem"):
        ChatRequest(messages=(user("hi"),), max_tokem=10)  # type: ignore[call-arg]


def test_chat_request_requires_at_least_one_message() -> None:
    with pytest.raises(ValidationError, match="messages"):
        ChatRequest(messages=())


def test_chat_request_optional_fields_default_to_none() -> None:
    request = ChatRequest(messages=(user("hi"),))

    assert request.system is None
    assert request.model is None
    assert request.max_output_tokens is None
    assert request.temperature is None


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_chat_request_rejects_temperature_out_of_range(temperature: float) -> None:
    with pytest.raises(ValidationError, match="temperature"):
        ChatRequest(messages=(user("hi"),), temperature=temperature)


def test_chat_requests_with_same_content_are_equal_and_hashable() -> None:
    # Igualdade por valor + hash: base para o cache de respostas (commit 18).
    first = ChatRequest(messages=(user("hi"),), temperature=0.0)
    second = ChatRequest(messages=(user("hi"),), temperature=0.0)

    assert first == second
    assert hash(first) == hash(second)


def test_usage_computes_total_tokens_and_serializes_it() -> None:
    usage = Usage(input_tokens=12, output_tokens=30)

    assert usage.total_tokens == 42
    assert usage.model_dump() == {"input_tokens": 12, "output_tokens": 30, "total_tokens": 42}


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1, output_tokens=0)


def test_chat_response_round_trips_through_json() -> None:
    response = ChatResponse(
        content="Olá!",
        model="model-x",
        provider="fake",
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=5, output_tokens=2),
    )

    restored = ChatResponse.model_validate_json(
        response.model_dump_json(exclude={"usage": {"total_tokens"}})
    )

    assert restored == response
