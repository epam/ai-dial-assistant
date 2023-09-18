from enum import Enum
from typing import Any

from aidial_sdk.chat_completion.request import Message, Role
from langchain.schema import BaseMessage, AIMessage, HumanMessage

from prompts.dialog import MAIN_SYSTEM_DIALOG_MESSAGE
from protocol.commands.base import commands_to_text
from protocol.commands.end_dialog import Reply


class StateField(str, Enum):
    INVOCATIONS = "invocations"
    REQUEST = "request"
    RESPONSE = "response"


class CommonField(str, Enum):
    INDEX = "index"


def sort_by_index(array: list[Any]):
    return array.sort(key=lambda item: int(item[CommonField.INDEX]))


def get_system_prefix(history: list[Message]) -> str:
    first_message = next(iter(history), None)
    if first_message is not None and first_message.role == Role.SYSTEM:
        return first_message.content

    return ""


def parse_history(
    history: list[Message],
    tools: dict[str, str],
) -> list[BaseMessage]:
    messages = [
        MAIN_SYSTEM_DIALOG_MESSAGE.format(
            system_prefix=get_system_prefix(history), tools=tools
        )
    ]

    for message in history:
        if message.role == Role.ASSISTANT:
            invocations = (
                message.custom_content.state.get(StateField.INVOCATIONS, [])
                if message.custom_content and message.custom_content.state
                else []
            )
            sort_by_index(invocations)
            for invocation in invocations:
                messages.append(AIMessage(content=invocation[StateField.REQUEST]))
                messages.append(HumanMessage(content=invocation[StateField.RESPONSE]))

            messages.append(
                AIMessage(
                    content=commands_to_text(
                        [
                            {
                                "command": Reply.token(),
                                "args": [message.content],
                            }
                        ]
                    )
                )
            )

        if message.role == Role.USER:
            messages.append(HumanMessage(content=message.content))

    return messages
