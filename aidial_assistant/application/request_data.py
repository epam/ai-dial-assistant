import json
from enum import Enum

from aidial_sdk.chat_completion import Addon, CustomContent, Message, Role
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from aidial_assistant.chain.command_result import (
    CommandInvocation,
    commands_to_text,
)
from aidial_assistant.utils.exceptions import RequestParameterValidationError
from aidial_assistant.utils.open_ai import (
    assistant_message,
    system_message,
    user_message,
)
from aidial_assistant.utils.open_ai_plugin import (
    OpenAIPluginInfo,
    get_open_ai_plugin_info,
)
from aidial_assistant.utils.state import Invocation, State


class AddonReference(BaseModel):
    name: str | None
    url: str


class PluginInfo(BaseModel):
    info: OpenAIPluginInfo
    url: str


class MessageScope(str, Enum):
    INTERNAL = "internal"  # internal dialog with plugins/addons, not visible to the user on the top level
    USER = "user"  # top-level dialog with the user


class ScopedMessage(BaseModel):
    scope: MessageScope = MessageScope.USER
    message: ChatCompletionMessageParam
    user_index: int


def _validate_messages(messages: list[Message]) -> None:
    if not messages:
        raise RequestParameterValidationError(
            "Message list cannot be empty.", param="messages"
        )

    if messages[-1].role != Role.USER:
        raise RequestParameterValidationError(
            "Last message must be from the user.", param="messages"
        )


def _validate_addons(addons: list[Addon] | None) -> list[AddonReference]:
    addon_references: list[AddonReference] = []
    for index, addon in enumerate(addons or []):
        if addon.url is None:
            raise RequestParameterValidationError(
                f"Missing required addon url at index {index}.",
                param="addons",
            )

        addon_references.append(AddonReference(name=addon.name, url=addon.url))

    return addon_references


def _get_model_args(request: ChatCompletionRequest) -> dict[str, str]:
    args = {
        "model": request.model,
        "temperature": request.temperature,
        "user": request.user,
    }

    return {k: v for k, v in args.items() if v is not None}


def _convert_old_commands(string: str) -> str:
    """Converts old commands to new format.
    Previously saved conversations with assistant will stop working if state is not updated.

    Old format:
    {"commands": [{"command": "run-addon", "args": ["<addon-name>", "<query>"]}]}
    New format:
    {"commands": [{"command": "<addon-name>", "arguments": {"query": "<query>"}}]}
    """
    commands = json.loads(string)
    result: list[CommandInvocation] = []

    for command in commands["commands"]:
        command_name = command["command"]
        # run-addon was previously called run-plugin
        if command_name in ("run-addon", "run-plugin"):
            args = command["args"]
            result.append(
                CommandInvocation(command=args[0], arguments={"query": args[1]})
            )
        else:
            result.append(command)

    return commands_to_text(result)


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


def _parse_history(history: list[Message]) -> list[ScopedMessage]:
    messages: list[ScopedMessage] = []
    for index, message in enumerate(history):
        if message.role == Role.ASSISTANT:
            invocations = _get_invocations(message.custom_content)
            for invocation in invocations:
                messages.append(
                    ScopedMessage(
                        scope=MessageScope.INTERNAL,
                        message=assistant_message(
                            _convert_old_commands(invocation["request"])
                        ),
                        user_index=index,
                    )
                )
                messages.append(
                    ScopedMessage(
                        scope=MessageScope.INTERNAL,
                        message=user_message(invocation["response"]),
                        user_index=index,
                    )
                )

            messages.append(
                ScopedMessage(
                    message=assistant_message(message.content or ""),
                    user_index=index,
                )
            )
        elif message.role == Role.USER:
            messages.append(
                ScopedMessage(
                    message=user_message(message.content or ""),
                    user_index=index,
                )
            )
        elif message.role == Role.SYSTEM:
            messages.append(
                ScopedMessage(
                    message=system_message(message.content or ""),
                    user_index=index,
                )
            )
        else:
            raise RequestParameterValidationError(
                f"Role {message.role} is not supported.", param="messages"
            )

    return messages


def get_discarded_user_messages(
    scoped_messages: list[ScopedMessage], discarded_messages: list[int]
) -> list[int]:
    return [scoped_messages[index].user_index for index in discarded_messages]


class RequestData(BaseModel):
    model_args: dict[str, str]
    messages: list[ScopedMessage]
    plugins: list[PluginInfo]
    addon_name_mapping: dict[str, str]
    max_prompt_tokens: int | None = None
    # TODO: Add max_addons_dialogue_tokens as a request parameter to the dial sdk
    max_addons_dialogue_tokens: int = 1000

    @classmethod
    async def from_dial_request(
        cls, request: ChatCompletionRequest
    ) -> "RequestData":
        _validate_messages(request.messages)
        addon_references = _validate_addons(request.addons)

        plugins: list[PluginInfo] = []
        # DIAL Core has own names for addons, so in stages we need to map them to the names used by the user
        addon_name_mapping: dict[str, str] = {}
        for addon_reference in addon_references:
            info = await get_open_ai_plugin_info(addon_reference.url)
            plugins.append(PluginInfo(info=info, url=addon_reference.url))

            if addon_reference.name:
                addon_name_mapping[
                    info.ai_plugin.name_for_model
                ] = addon_reference.name

        return cls(
            model_args=_get_model_args(request),
            messages=_parse_history(request.messages),
            plugins=plugins,
            addon_name_mapping=addon_name_mapping,
            max_prompt_tokens=request.max_prompt_tokens,
        )
