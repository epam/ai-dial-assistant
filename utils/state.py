from enum import Enum
from typing import Any

from langchain.schema import BaseMessage, AIMessage, HumanMessage

from conf.project_conf import PluginOpenAI
from prompts.dialog import MAIN_SYSTEM_DIALOG_MESSAGE, RESP_DIALOG_PROMPT
from protocol.command_result import responses_to_text, Status
from protocol.commands.base import commands_to_text
from protocol.commands.end_dialog import Reply
from protocol.commands.say_or_ask import SayOrAsk


class OpenAIRole(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class MessageField(str, Enum):
    ROLE = "role"
    CUSTOM_CONTENT = "custom_content"


class CustomContentField(str, Enum):
    STAGES = "stages"
    STATE = "state"


class StageField(str, Enum):
    STATUS = "status"
    NAME = "name"


class StageStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class StateField(str, Enum):
    INVOCATIONS = "invocations"
    REQUEST = "request"
    RESPONSE = "response"


class CommonField(str, Enum):
    INDEX = "index"
    CONTENT = "content"


def sort_by_index(array: list[Any]):
    return array.sort(key=lambda item: int(item[CommonField.INDEX]))


def get_system_prefix(history: list[Any]) -> str:
    first_message = next(iter(history), None)
    if first_message is not None and first_message[MessageField.ROLE] == OpenAIRole.SYSTEM:
        return first_message[CommonField.CONTENT]

    return ""


def parse_history(
    history: list[Any],
    tools: dict[str, str],
) -> list[BaseMessage]:
    messages = [
        MAIN_SYSTEM_DIALOG_MESSAGE.format(system_prefix=get_system_prefix(history), tools=tools)
    ]

    for message in history:
        if message["role"] == OpenAIRole.ASSISTANT:
            invocations = message.get(MessageField.CUSTOM_CONTENT, {})\
                .get(CustomContentField.STATE, {})\
                .get(StateField.INVOCATIONS, [])
            sort_by_index(invocations)
            for invocation in invocations:
                messages.append(AIMessage(content=invocation[StateField.REQUEST]))
                messages.append(HumanMessage(content=invocation[StateField.RESPONSE]))

            messages.append(AIMessage(content=commands_to_text(
                [{"command": Reply.token(), "args": [message[CommonField.CONTENT]]}]
            )))

        if message[MessageField.ROLE] == OpenAIRole.USER:
            messages.append(HumanMessage(content=message[CommonField.CONTENT]))

    return messages
