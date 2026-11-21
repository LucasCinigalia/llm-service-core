"""O port é verificado pelo mypy, não em tempo de execução.

``EchoProvider`` não herda de ``LLMProvider``. Mesmo assim, atribuí-lo a uma variável do tipo
``LLMProvider`` passa no mypy porque ele tem a mesma "forma" (tipagem estrutural / duck typing
estático). Se você remover ``name`` ou mudar a assinatura de ``chat``, o ``make typecheck`` falha.
"""

from collections.abc import AsyncGenerator

from app.domain.llm import (
    ChatRequest,
    ChatResponse,
    Message,
    Role,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    Usage,
)
from app.domain.ports import LLMProvider


class EchoProvider:
    @property
    def name(self) -> str:
        return "echo"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        last = request.messages[-1].content
        return ChatResponse(
            content=last,
            model="echo-1",
            provider=self.name,
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=len(last), output_tokens=len(last)),
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        response = await self.chat(request)
        yield TextDelta(text=response.content)
        yield StreamCompleted(response=response)

    async def aclose(self) -> None:
        pass


async def ask(provider: LLMProvider, text: str) -> str:
    """Código que depende só do port — não sabe qual provider recebeu."""
    response = await provider.chat(ChatRequest(messages=(Message(role=Role.USER, content=text),)))
    return response.content


async def test_any_class_with_the_right_shape_is_a_provider() -> None:
    provider: LLMProvider = EchoProvider()

    assert await ask(provider, "ping") == "ping"
    assert provider.name == "echo"
