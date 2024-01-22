import logging
from pathlib import Path
from typing import Callable, Tuple

from aidial_sdk.chat_completion import FinishReason
from aidial_sdk.chat_completion.base import ChatCompletion
from aidial_sdk.chat_completion.request import Request
from aidial_sdk.chat_completion.response import Response
from aidial_sdk.deployment.tokenize import (
    TokenizeError,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeSuccess,
)
from aidial_sdk.deployment.truncate_prompt import (
    TruncatePromptError,
    TruncatePromptRequest,
    TruncatePromptResponse,
    TruncatePromptSuccess,
)
from openai.lib.azure import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionToolParam
from typing_extensions import override

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
from aidial_assistant.application.request_data import (
    PluginInfo,
    RequestData,
    get_discarded_user_messages,
)
from aidial_assistant.chain.command_chain import (
    CommandChain,
    CommandConstructor,
    CommandDict,
)
from aidial_assistant.chain.history import History, ScopedMessage
from aidial_assistant.commands.reply import Reply
from aidial_assistant.commands.run_plugin import RunPlugin
from aidial_assistant.commands.run_tool import RunTool
from aidial_assistant.model.model_client import (
    ModelClient,
    ModelClientRequest,
    ReasonLengthException,
)
from aidial_assistant.model.tokenize_client import (
    TokenizeClient,
    TokenizeClientRequest,
)
from aidial_assistant.model.truncate_propmt_client import (
    TruncatePromptClient,
    TruncatePromptClientRequest,
)
from aidial_assistant.tools_chain.tools_chain import (
    CommandToolDict,
    ToolsChain,
    convert_commands_to_tools,
)
from aidial_assistant.utils.exceptions import (
    RequestParameterValidationError,
    UnauthorizedAddonError,
    unhandled_exception_handler,
)
from aidial_assistant.utils.open_ai import construct_tool
from aidial_assistant.utils.open_ai_plugin import AddonTokenSource, AIPluginConf
from aidial_assistant.utils.state import State

logger = logging.getLogger(__name__)


def _construct_tool(plugin_conf: AIPluginConf) -> ChatCompletionToolParam:
    return construct_tool(
        plugin_conf.name_for_model,
        plugin_conf.description_for_human,
        {
            "query": {
                "type": "string",
                "description": "A task written in natural language",
            }
        },
        ["query"],
    )


def _create_history(
    messages: list[ScopedMessage], plugins: list[PluginInfo]
) -> History:
    plugin_descriptions = {
        plugin.info.ai_plugin.name_for_model: plugin.info.open_api.info.description
        or plugin.info.ai_plugin.description_for_human
        for plugin in plugins
    }
    return History(
        assistant_system_message_template=MAIN_SYSTEM_DIALOG_MESSAGE.build(
            addons=plugin_descriptions
        ),
        best_effort_template=MAIN_BEST_EFFORT_TEMPLATE.build(
            addons=plugin_descriptions
        ),
        scoped_messages=messages,
    )


def _get_plugin_auth(
    plugin: PluginInfo, token_source: AddonTokenSource
) -> str | None:
    auth_type = plugin.info.ai_plugin.auth.type

    if auth_type == "none":
        return token_source.default_auth

    if auth_type == "service_http":
        service_token = token_source.get_token(plugin.url)
        if service_token is None:
            raise UnauthorizedAddonError(f"Missing token for {plugin.url}")

        authorization_type = plugin.info.ai_plugin.auth.authorization_type

        # Capitalizing because Wolfram, for instance, doesn't like lowercase bearer
        return f"{authorization_type.capitalize()} {service_token}"

    raise UnauthorizedAddonError(f"Unknown auth type {auth_type}")


def _native_tools_request(request_data: RequestData) -> ModelClientRequest:
    tools = [
        _construct_tool(plugin.info.ai_plugin)
        for plugin in request_data.plugins
    ]
    return ModelClientRequest(
        messages=convert_commands_to_tools(request_data.messages),
        max_prompt_tokens=request_data.max_prompt_tokens,
        tools=tools,
    )


def _emulated_tools_request(request_data: RequestData) -> ModelClientRequest:
    history = _create_history(request_data.messages, request_data.plugins)
    return ModelClientRequest(
        messages=history.to_protocol_messages(),
        max_prompt_tokens=request_data.max_prompt_tokens,
    )


class AssistantApplication(ChatCompletion):
    def __init__(
        self, config_dir: Path, tools_supporting_deployments: set[str]
    ):
        self.args = parse_args(config_dir)
        self.tools_supporting_deployments = tools_supporting_deployments

    @override
    @unhandled_exception_handler
    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        request_data = await RequestData.from_dial_request(request)

        model = ModelClient(
            client=AsyncAzureOpenAI(
                azure_endpoint=self.args.openai_conf.api_base,
                api_key=request.api_key,
                # 2023-12-01-preview is needed to support tools
                api_version="2023-12-01-preview",
            ),
            model_args=request_data.model_args,
        )
        token_source = AddonTokenSource(
            request.headers,
            (plugin.url for plugin in request_data.plugins),
        )

        if self._supports_native_tools(request.model):
            await AssistantApplication._run_native_tools_chat(
                model, token_source, request_data, response
            )
        else:
            await AssistantApplication._run_emulated_tools_chat(
                model, token_source, request_data, response
            )

    @staticmethod
    async def _run_emulated_tools_chat(
        model: ModelClient,
        token_source: AddonTokenSource,
        request_data: RequestData,
        response: Response,
    ):
        def create_command(plugin: PluginInfo, auth: str | None):
            return lambda: RunPlugin(
                model, plugin, auth, request_data.max_addons_dialogue_tokens
            )

        command_dict: CommandDict = {
            plugin.info.ai_plugin.name_for_model: create_command(
                plugin, _get_plugin_auth(plugin, token_source)
            )
            for plugin in request_data.plugins
        }
        if Reply.token() in command_dict:
            RequestParameterValidationError(
                f"Addon with name '{Reply.token()}' is not allowed in emulated tools mode.",
                param="addons",
            )

        command_dict[Reply.token()] = Reply

        chain = CommandChain(
            model_client=model, name="ASSISTANT", command_dict=command_dict
        )
        history = _create_history(request_data.messages, request_data.plugins)
        discarded_user_messages: list[int] | None = None
        if request_data.max_prompt_tokens is not None:
            history, discarded_messages = await history.truncate(
                model, request_data.max_prompt_tokens
            )
            discarded_user_messages = get_discarded_user_messages(
                request_data.messages, discarded_messages
            )
        # TODO: else compare the history size to the max prompt tokens of the underlying model

        choice = response.create_single_choice()
        choice.open()

        callback = AssistantChainCallback(
            choice, request_data.addon_name_mapping
        )
        finish_reason = FinishReason.STOP
        try:
            model_request_limiter = AddonsDialogueLimiter(
                request_data.max_addons_dialogue_tokens, model
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

        if discarded_user_messages is not None:
            response.set_discarded_messages(discarded_user_messages)

    @staticmethod
    async def _run_native_tools_chat(
        model: ModelClient,
        token_source: AddonTokenSource,
        request_data: RequestData,
        response: Response,
    ):
        def create_command_tool(
            plugin: PluginInfo, auth: str | None
        ) -> Tuple[CommandConstructor, ChatCompletionToolParam]:
            return lambda: RunTool(
                model, plugin, auth, request_data.max_addons_dialogue_tokens
            ), _construct_tool(plugin.info.ai_plugin)

        commands: CommandToolDict = {
            plugin.info.ai_plugin.name_for_model: create_command_tool(
                plugin, _get_plugin_auth(plugin, token_source)
            )
            for plugin in request_data.plugins
        }
        chain = ToolsChain(model, commands)

        choice = response.create_single_choice()
        choice.open()

        callback = AssistantChainCallback(
            choice, request_data.addon_name_mapping
        )
        finish_reason = FinishReason.STOP
        messages = convert_commands_to_tools(request_data.messages)
        try:
            model_request_limiter = AddonsDialogueLimiter(
                request_data.max_addons_dialogue_tokens, model
            )
            await chain.run_chat(messages, callback, model_request_limiter)
        except ReasonLengthException:
            finish_reason = FinishReason.LENGTH

        if callback.invocations:
            choice.set_state(State(invocations=callback.invocations))
        choice.close(finish_reason)

        response.set_usage(
            model.total_prompt_tokens, model.total_completion_tokens
        )

    @override
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        inputs: list[ModelClientRequest | str] = []

        for tokenizer_input in request.inputs:
            if tokenizer_input.type == "string":
                inputs.append(tokenizer_input.value)
                continue

            request_data = await RequestData.from_dial_request(
                tokenizer_input.value
            )
            if self._supports_native_tools(tokenizer_input.value.model):
                inputs.append(_native_tools_request(request_data))
            else:
                inputs.append(_emulated_tools_request(request_data))

        client = TokenizeClient(self.args.openai_conf.api_base)
        outputs = await client.tokenize(TokenizeClientRequest(inputs=inputs))

        return TokenizeResponse(
            outputs=[
                TokenizeSuccess(token_count=output)
                if isinstance(output, int)
                else TokenizeError(error=output)
                for index, output in enumerate(outputs)
            ]
        )

    @override
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        inputs: list[ModelClientRequest] = []
        indices_converters: list[Callable[[list[int]], list[int]]] = []

        for completion_request in request.inputs:
            request_data = await RequestData.from_dial_request(
                completion_request
            )
            if self._supports_native_tools(completion_request.model):
                inputs.append(_native_tools_request(request_data))
                indices_converters.append(lambda indices: indices)
            else:
                inputs.append(_emulated_tools_request(request_data))
                indices_converters.append(
                    lambda indices: get_discarded_user_messages(
                        request_data.messages, indices
                    )
                )

        client = TruncatePromptClient(self.args.openai_conf.api_base)
        outputs = await client.truncate_prompt(
            TruncatePromptClientRequest(inputs=inputs)
        )

        return TruncatePromptResponse(
            outputs=[
                TruncatePromptSuccess(
                    discarded_messages=indices_converters[index](output)
                )
                if isinstance(output, list)
                else TruncatePromptError(error=output)
                for index, output in enumerate(outputs)
            ]
        )

    def _supports_native_tools(self, model: str) -> bool:
        return model in self.tools_supporting_deployments
