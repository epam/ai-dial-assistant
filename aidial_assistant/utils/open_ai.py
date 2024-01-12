from typing import TypedDict

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class Property(TypedDict, total=False):
    type: str
    description: str


def construct_tool(
    name: str,
    description: str,
    properties: dict[str, Property],
    required: list[str],
) -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        ),
    )


def system_message(content: str) -> ChatCompletionSystemMessageParam:
    return ChatCompletionSystemMessageParam(role="system", content=content)


def user_message(content: str) -> ChatCompletionUserMessageParam:
    return ChatCompletionUserMessageParam(role="user", content=content)


def assistant_message(content: str) -> ChatCompletionAssistantMessageParam:
    return ChatCompletionAssistantMessageParam(
        role="assistant", content=content
    )


def tool_calls_message(
    tool_calls: list[ChatCompletionMessageToolCallParam],
) -> ChatCompletionAssistantMessageParam:
    return ChatCompletionAssistantMessageParam(
        role="assistant", tool_calls=tool_calls
    )


def tool_message(
    content: str, tool_call_id: str
) -> ChatCompletionToolMessageParam:
    return ChatCompletionToolMessageParam(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
    )
