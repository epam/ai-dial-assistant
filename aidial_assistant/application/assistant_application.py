import json
import logging
from pathlib import Path

from aidial_sdk.chat_completion import FinishReason
from aidial_sdk.chat_completion.base import ChatCompletion
from aidial_sdk.chat_completion.request import Addon
from aidial_sdk.chat_completion.request import Message as SdkMessage
from aidial_sdk.chat_completion.request import Request, Role
from aidial_sdk.chat_completion.response import Response
from openai.lib.azure import AsyncAzureOpenAI
from pydantic import BaseModel

from aidial_assistant.application.addons_dialogue_limiter import (
    AddonsDialogueLimiter,
)
from aidial_assistant.application.args import parse_args
from aidial_assistant.application.assistant_callback import (
    AssistantChainCallback,
)
from aidial_assistant.application.prompts import (
    MAIN_BEST_EFFORT_TEMPLATE,
    MAIN_SYSTEM_DIALOG_MESSAGE,
)
from aidial_assistant.chain.command_chain import CommandChain, CommandDict
from aidial_assistant.chain.command_result import Commands, Responses
from aidial_assistant.chain.history import History, MessageScope, ScopedMessage
from aidial_assistant.commands.reply import Reply
from aidial_assistant.commands.run_plugin import PluginInfo, RunPlugin
from aidial_assistant.commands.run_tool import RunTool
from aidial_assistant.model.model_client import (
    Message,
    ModelClient,
    ReasonLengthException,
    ToolCall,
)
from aidial_assistant.tools_chain.tools_chain import ToolsChain
from aidial_assistant.utils.exceptions import (
    RequestParameterValidationError,
    unhandled_exception_handler,
)
from aidial_assistant.utils.open_ai import (
    FunctionCall,
    Tool,
    construct_function,
)
from aidial_assistant.utils.open_ai_plugin import (
    AddonTokenSource,
    get_open_ai_plugin_info,
    get_plugin_auth,
)
from aidial_assistant.utils.state import State, parse_history

logger = logging.getLogger(__name__)


class AddonReference(BaseModel):
    name: str | None
    url: str


def _get_request_args(request: Request) -> dict[str, str]:
    args = {
        "model": request.model,
        "temperature": request.temperature,
        "user": request.user,
    }

    return {k: v for k, v in args.items() if v is not None}


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


def _validate_messages(messages: list[SdkMessage]) -> None:
    if not messages:
        raise RequestParameterValidationError(
            "Message list cannot be empty.", param="messages"
        )

    if messages[-1].role != Role.USER:
        raise RequestParameterValidationError(
            "Last message must be from the user.", param="messages"
        )


def _validate_request(request: Request) -> None:
    _validate_messages(request.messages)
    _validate_addons(request.addons)


def _construct_function(name: str, description: str) -> Tool:
    return construct_function(
        name,
        description,
        {
            "query": {
                "type": "string",
                "description": "A task written in natural language",
            }
        },
        ["query"],
    )


def _convert_commands_to_tools(
    scoped_messages: list[ScopedMessage],
) -> list[Message]:
    messages: list[Message] = []
    next_tool_id: int = 0
    last_call_count: int = 0
    for scoped_message in scoped_messages:
        message = scoped_message.message
        if scoped_message.scope == MessageScope.INTERNAL:
            if not message.content:
                raise RequestParameterValidationError(
                    "State is broken. Content cannot be empty.",
                    param="messages",
                )

            if message.role == Role.ASSISTANT:
                commands: Commands = json.loads(message.content)
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=[
                            ToolCall(
                                index=index,
                                id=str(next_tool_id + index),
                                function=FunctionCall(
                                    name=command["command"],
                                    arguments=json.dumps(command["arguments"]),
                                ),
                                type="function",
                            )
                            for index, command in enumerate(
                                commands["commands"]
                            )
                        ],
                    )
                )
                last_call_count = len(commands["commands"])
                next_tool_id += last_call_count
            elif message.role == Role.USER:
                responses: Responses = json.loads(message.content)
                response_count = len(responses["responses"])
                if response_count != last_call_count:
                    raise RequestParameterValidationError(
                        f"Expected {last_call_count} responses, but got {response_count}.",
                        param="messages",
                    )
                first_tool_id = next_tool_id - last_call_count
                messages.extend(
                    [
                        Message(
                            role=Role.TOOL,
                            tool_call_id=str(first_tool_id + index),
                            content=response["response"],
                        )
                        for index, response in enumerate(responses["responses"])
                    ]
                )
        else:
            messages.append(scoped_message.message)
    return messages


class AssistantApplication(ChatCompletion):
    def __init__(self, config_dir: Path):
        self.args = parse_args(config_dir)

    @unhandled_exception_handler
    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        _validate_messages(request.messages)
        addon_references = _validate_addons(request.addons)
        chat_args = _get_request_args(request)

        model = ModelClient(
            client=AsyncAzureOpenAI(
                azure_endpoint=self.args.openai_conf.api_base,
                api_key=request.api_key,
                api_version=request.api_version,
            ),
            model_args=chat_args,
        )

        token_source = AddonTokenSource(
            request.headers,
            (addon_reference.url for addon_reference in addon_references),
        )

        addons: list[PluginInfo] = []
        # DIAL Core has own names for addons, so in stages we need to map them to the names used by the user
        addon_name_mapping: dict[str, str] = {}
        for addon_reference in addon_references:
            info = await get_open_ai_plugin_info(addon_reference.url)
            addons.append(
                PluginInfo(
                    info=info,
                    auth=get_plugin_auth(
                        info.ai_plugin.auth.type,
                        info.ai_plugin.auth.authorization_type,
                        addon_reference.url,
                        token_source,
                    ),
                )
            )

            if addon_reference.name:
                addon_name_mapping[
                    info.ai_plugin.name_for_model
                ] = addon_reference.name

        if request.model in {"gpt-4-turbo-1106", "anthropic.claude-v2-1"}:
            await AssistantApplication._run_native_tools_chat(
                model, addons, addon_name_mapping, request, response
            )
        else:
            await AssistantApplication._run_emulated_tools_chat(
                model, addons, addon_name_mapping, request, response
            )

    @staticmethod
    async def _run_emulated_tools_chat(
        model: ModelClient,
        addons: list[PluginInfo],
        addon_name_mapping: dict[str, str],
        request: Request,
        response: Response,
    ):
        # TODO: Add max_addons_dialogue_tokens as a request parameter
        max_addons_dialogue_tokens = 1000

        def create_command(addon: PluginInfo):
            return lambda: RunPlugin(model, addon, max_addons_dialogue_tokens)

        command_dict: CommandDict = {
            addon.info.ai_plugin.name_for_model: create_command(addon)
            for addon in addons
        }
        command_dict[Reply.token()] = Reply

        chain = CommandChain(
            model_client=model, name="ASSISTANT", command_dict=command_dict
        )
        addon_descriptions = {
            addon.info.ai_plugin.name_for_model: addon.info.open_api.info.description
            or addon.info.ai_plugin.description_for_human
            for addon in addons
        }
        history = History(
            assistant_system_message_template=MAIN_SYSTEM_DIALOG_MESSAGE.build(
                addons=addon_descriptions
            ),
            best_effort_template=MAIN_BEST_EFFORT_TEMPLATE.build(
                addons=addon_descriptions
            ),
            scoped_messages=parse_history(request.messages),
        )
        discarded_messages: int | None = None
        if request.max_prompt_tokens is not None:
            original_size = history.user_message_count
            history = await history.truncate(request.max_prompt_tokens, model)
            truncated_size = history.user_message_count
            discarded_messages = original_size - truncated_size
        # TODO: else compare the history size to the max prompt tokens of the underlying model

        choice = response.create_single_choice()
        choice.open()

        callback = AssistantChainCallback(choice, addon_name_mapping)
        finish_reason = FinishReason.STOP
        try:
            model_request_limiter = AddonsDialogueLimiter(
                max_addons_dialogue_tokens, model
            )
            await chain.run_chat(history, callback, model_request_limiter)
        except ReasonLengthException:
            finish_reason = FinishReason.LENGTH

        if callback.invocations:
            choice.set_state(State(invocations=callback.invocations))

        choice.close(finish_reason)

        response.set_usage(
            model.total_prompt_tokens, model.total_completion_tokens
        )

        if discarded_messages is not None:
            response.set_discarded_messages(discarded_messages)

    @staticmethod
    async def _run_native_tools_chat(
        model: ModelClient,
        addons: list[PluginInfo],
        addon_name_mapping: dict[str, str],
        request: Request,
        response: Response,
    ):
        tools: list[Tool] = [
            _construct_function(
                addon.info.ai_plugin.name_for_model,
                addon.info.ai_plugin.description_for_human,
            )
            for addon in addons
        ]

        def create_command(addon: PluginInfo):
            return lambda: RunTool(model, addon)

        command_dict: CommandDict = {
            addon.info.ai_plugin.name_for_model: create_command(addon)
            for addon in addons
        }
        chain = ToolsChain(model, tools, command_dict)

        choice = response.create_single_choice()
        choice.open()

        callback = AssistantChainCallback(choice, addon_name_mapping)
        finish_reason = FinishReason.STOP
        messages = _convert_commands_to_tools(parse_history(request.messages))
        try:
            await chain.run_chat(messages, callback)
        except ReasonLengthException:
            finish_reason = FinishReason.LENGTH

        if callback.invocations:
            choice.set_state(State(invocations=callback.invocations))
        choice.close(finish_reason)

        response.set_usage(
            model.total_prompt_tokens, model.total_completion_tokens
        )
