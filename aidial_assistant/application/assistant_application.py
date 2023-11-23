import logging
from pathlib import Path

from aidial_sdk.chat_completion import FinishReason
from aidial_sdk.chat_completion.base import ChatCompletion
from aidial_sdk.chat_completion.request import Addon, Message, Request, Role
from aidial_sdk.chat_completion.response import Response
from aiohttp import hdrs

from aidial_assistant.application.args import parse_args
from aidial_assistant.application.assistant_callback import (
    AssistantChainCallback,
)
from aidial_assistant.application.prompts import (
    MAIN_BEST_EFFORT_TEMPLATE,
    MAIN_SYSTEM_DIALOG_MESSAGE,
)
from aidial_assistant.chain.command_chain import CommandChain, CommandDict
from aidial_assistant.chain.history import History
from aidial_assistant.chain.model_client import (
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.commands.reply import Reply
from aidial_assistant.commands.run_plugin import PluginInfo, RunPlugin
from aidial_assistant.utils.exceptions import (
    RequestParameterValidationError,
    unhandled_exception_handler,
)
from aidial_assistant.utils.open_ai_plugin import (
    AddonTokenSource,
    get_open_ai_plugin_info,
    get_plugin_auth,
)
from aidial_assistant.utils.state import State, parse_history

logger = logging.getLogger(__name__)


def _get_request_args(request: Request) -> dict[str, str]:
    args = {
        "model": request.model,
        "temperature": request.temperature,
        "api_version": request.api_version,
        "api_key": request.api_key,
        "user": request.user,
        "headers": None
        if request.jwt is None
        else {hdrs.AUTHORIZATION: request.jwt},
    }

    return {k: v for k, v in args.items() if v is not None}


def _extract_addon_url(addon: Addon) -> str:
    if addon.url is None:
        raise RequestParameterValidationError(
            "Missing required addon url.",
            param="addons",
        )

    return addon.url


def _validate_messages(messages: list[Message]) -> None:
    if messages[-1].role != Role.USER:
        raise RequestParameterValidationError(
            "Last message must be from the user.", param="messages"
        )


class AssistantApplication(ChatCompletion):
    def __init__(self, config_dir: Path):
        self.args = parse_args(config_dir)

    @unhandled_exception_handler
    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        chat_args = self.args.openai_conf.dict() | _get_request_args(request)

        model = ModelClient(
            model_args=chat_args
            | {
                "deployment_id": chat_args["model"],
                "api_type": "azure",
                "stream": True,
            },
            buffer_size=self.args.chat_conf.buffer_size,
        )

        addons = (
            [_extract_addon_url(addon) for addon in request.addons]
            if request.addons
            else []
        )
        token_source = AddonTokenSource(request.headers, addons)

        tools: dict[str, PluginInfo] = {}
        tool_descriptions: dict[str, str] = {}
        for addon in addons:
            info = await get_open_ai_plugin_info(addon)
            tools[info.ai_plugin.name_for_model] = PluginInfo(
                info=info,
                auth=get_plugin_auth(
                    info.ai_plugin.auth.type,
                    info.ai_plugin.auth.authorization_type,
                    addon,
                    token_source,
                ),
            )

            tool_descriptions[info.ai_plugin.name_for_model] = (
                info.open_api.info.description  # type: ignore
                or info.ai_plugin.description_for_human
            )

        command_dict: CommandDict = {
            RunPlugin.token(): lambda: RunPlugin(model, tools),
            Reply.token(): Reply,
        }
        chain = CommandChain(
            model_client=model, name="ASSISTANT", command_dict=command_dict
        )
        _validate_messages(request.messages)
        history = History(
            assistant_system_message_template=MAIN_SYSTEM_DIALOG_MESSAGE.build(
                tools=tool_descriptions
            ),
            best_effort_template=MAIN_BEST_EFFORT_TEMPLATE.build(
                tools=tool_descriptions
            ),
            scoped_messages=parse_history(request.messages),
        )
        discarded_messages: int | None = None
        if request.max_prompt_tokens is not None:
            old_size = history.user_message_count()
            history = await history.trim(request.max_prompt_tokens, model)
            discarded_messages = old_size - history.user_message_count()

        choice = response.create_single_choice()
        choice.open()

        callback = AssistantChainCallback(choice)
        finish_reason = FinishReason.STOP
        try:
            await chain.run_chat(history, callback)
        except ReasonLengthException:
            finish_reason = FinishReason.LENGTH

        if callback.invocations:
            choice.set_state(State(invocations=callback.invocations))

        choice.close(finish_reason)

        response.set_usage(model.prompt_tokens, model.completion_tokens)

        if discarded_messages:
            response.set_discarded_messages(discarded_messages)
