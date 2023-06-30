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


def sort_by_index(array: list[Any]):
    return array.sort(key=lambda item: int(item["index"]))


def get_system_prefix(history: list[Any]) -> str:
    first_message = next(iter(history), None)
    if first_message is not None and first_message["role"] == OpenAIRole.SYSTEM:
        return first_message["content"]

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
            invocations = message.get("custom_content", {})\
                .get("state", {})\
                .get("invocations", [])
            sort_by_index(invocations)
            for invocation in invocations:
                messages.append(AIMessage(content=invocation["request"]))
                messages.append(RESP_DIALOG_PROMPT.format(responses=invocation["response"]))

            messages.append(AIMessage(content=commands_to_text(
                [{"command": SayOrAsk.token(), "args": [message["content"]]}]
            )))

        if message["role"] == OpenAIRole.USER:
            responses = responses_to_text([{"status": Status.SUCCESS, "response": message["content"]}])
            messages.append(RESP_DIALOG_PROMPT.format(responses=responses))

    return messages
