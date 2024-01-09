from typing import Any, TypedDict


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class Property(TypedDict, total=False):
    type: str
    description: str
    default: Any


class Parameters(TypedDict):
    type: str
    properties: dict[str, Property]
    required: list[str]


class Function(TypedDict):
    name: str
    description: str
    parameters: Parameters


class Tool(TypedDict):
    type: str
    function: Function


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    index: int
    id: str
    type: str
    function: FunctionCall


def construct_function(
    name: str,
    description: str,
    properties: dict[str, Property],
    required: list[str],
) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
