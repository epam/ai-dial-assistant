import json
from enum import Enum
from typing import Any

from aidial_sdk.chat_completion.request import Message, Role

from aidial_assistant.commands.reply import Reply


class StateField(str, Enum):
    INDEX = "index"
    INVOCATIONS = "invocations"
    REQUEST = "request"
    RESPONSE = "response"


def sort_by_index(array: list[Any]):
    return array.sort(key=lambda item: int(item[StateField.INDEX]))


def get_system_prefix(history: list[Message]) -> str:
    first_message = next(iter(history), None)
    if first_message is not None and first_message.role == Role.SYSTEM:
        return first_message.content or ""

    return ""


def parse_history(
    history: list[Message],
    system_message: str,
) -> list[Message]:
    messages = [Message(role=Role.SYSTEM, content=system_message)]

    for message in history:
        if message.role == Role.ASSISTANT:
            invocations = (
                message.custom_content.state.get(StateField.INVOCATIONS, [])
                if message.custom_content and message.custom_content.state
                else []
            )
            sort_by_index(invocations)
            for invocation in invocations:
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=invocation[StateField.REQUEST],
                    )
                )
                messages.append(
                    Message(
                        role=Role.USER, content=invocation[StateField.RESPONSE]
                    )
                )

            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=json.dumps(
                        {
                            "commands": [
                                {
                                    "command": Reply.token(),
                                    "args": [message.content or ""],
                                }
                            ]
                        }
                    ),
                )
            )

        if message.role == Role.USER:
            messages.append(message)

    return messages
