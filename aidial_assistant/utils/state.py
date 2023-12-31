from typing import TypedDict

from aidial_sdk.chat_completion.request import CustomContent, Message, Role

from aidial_assistant.chain.history import MessageScope, ScopedMessage
from aidial_assistant.model.model_client import Message as ModelMessage


class Invocation(TypedDict):
    index: str | int
    request: str
    response: str


class State(TypedDict, total=False):
    invocations: list[Invocation]


def _get_invocations(custom_content: CustomContent | None) -> list[Invocation]:
    if custom_content is None:
        return []

    state: State | None = custom_content.state
    if state is None:
        return []

    invocations: list[Invocation] | None = state.get("invocations")
    if invocations is None:
        return []

    invocations.sort(key=lambda invocation: int(invocation["index"]))
    return invocations


def parse_history(history: list[Message]) -> list[ScopedMessage]:
    messages: list[ScopedMessage] = []
    for message in history:
        if message.role == Role.ASSISTANT:
            invocations = _get_invocations(message.custom_content)
            for invocation in invocations:
                messages.append(
                    ScopedMessage(
                        scope=MessageScope.INTERNAL,
                        message=ModelMessage.assistant(invocation["request"]),
                    )
                )
                messages.append(
                    ScopedMessage(
                        scope=MessageScope.INTERNAL,
                        message=ModelMessage.user(invocation["response"]),
                    )
                )

        messages.append(
            ScopedMessage(
                message=ModelMessage(
                    role=message.role, content=message.content or ""
                )
            )
        )

    return messages
