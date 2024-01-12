import json
from typing import TypedDict

from aidial_sdk.chat_completion.request import CustomContent, Message, Role

from aidial_assistant.chain.command_result import (
    CommandInvocation,
    commands_to_text,
)
from aidial_assistant.chain.history import MessageScope, ScopedMessage
from aidial_assistant.utils.exceptions import RequestParameterValidationError
from aidial_assistant.utils.open_ai import (
    assistant_message,
    system_message,
    user_message,
)


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


def _convert_old_commands(string: str) -> str:
    """Converts old commands to new format."""
    commands = json.loads(string)
    result: list[CommandInvocation] = []

    for command in commands["commands"]:
        command_name = command["command"]
        if command_name in ("run-addon", "run-plugin"):
            args = command["args"]
            result.append(
                CommandInvocation(command=args[0], arguments={"query": args[1]})
            )
        else:
            result.append(command)

    return commands_to_text(result)


def parse_history(history: list[Message]) -> list[ScopedMessage]:
    messages: list[ScopedMessage] = []
    for message in history:
        if message.role == Role.ASSISTANT:
            invocations = _get_invocations(message.custom_content)
            for invocation in invocations:
                messages.append(
                    ScopedMessage(
                        scope=MessageScope.INTERNAL,
                        message=assistant_message(
                            _convert_old_commands(invocation["request"])
                        ),
                    )
                )
                messages.append(
                    ScopedMessage(
                        scope=MessageScope.INTERNAL,
                        message=user_message(invocation["response"]),
                    )
                )

            messages.append(
                ScopedMessage(message=assistant_message(message.content or ""))
            )
        elif message.role == Role.USER:
            messages.append(
                ScopedMessage(message=user_message(message.content or ""))
            )
        elif message.role == Role.SYSTEM:
            messages.append(
                ScopedMessage(message=system_message(message.content or ""))
            )
        else:
            raise RequestParameterValidationError(
                f"Role {message.role} is not supported.", param="messages"
            )

    return messages
