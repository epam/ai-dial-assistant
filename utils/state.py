from enum import Enum
from typing import Any

from langchain.schema import BaseMessage, AIMessage

from conf.project_conf import PluginOpenAI
from prompts.dialog import SYSTEM_DIALOG_MESSAGE, RESP_DIALOG_PROMPT
from protocol.command_result import responses_to_text, Status
from protocol.commands.base import commands_to_text
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
        SYSTEM_DIALOG_MESSAGE.format(system_prefix=get_system_prefix(history), commands={}, tools=tools),
        AIMessage(content=commands_to_text([{"command": SayOrAsk.token(), "args": ["How can I help you?"]}]))
    ]

    for message in history:
        if message["role"] == OpenAIRole.ASSISTANT:
            invocations = message.get(MessageField.CUSTOM_CONTENT, {})\
                .get(CustomContentField.STATE, {})\
                .get(StateField.INVOCATIONS, [])
            sort_by_index(invocations)
            for invocation in invocations:
                messages.append(AIMessage(content=invocation[StateField.REQUEST]))
                messages.append(RESP_DIALOG_PROMPT.format(responses=invocation[StateField.RESPONSE]))

            messages.append(AIMessage(content=commands_to_text(
                [{"command": SayOrAsk.token(), "args": [message[CommonField.CONTENT]]}]
            )))

        if message[MessageField.ROLE] == OpenAIRole.USER:
            responses = responses_to_text([{"status": Status.SUCCESS, "response": message[CommonField.CONTENT]}])
            messages.append(RESP_DIALOG_PROMPT.format(responses=responses))

    return messages
