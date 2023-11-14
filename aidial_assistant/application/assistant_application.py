import logging
from pathlib import Path

from aidial_sdk import HTTPException
from aidial_sdk.chat_completion import FinishReason
from aidial_sdk.chat_completion.base import ChatCompletion
from aidial_sdk.chat_completion.request import Addon, Request
from aidial_sdk.chat_completion.response import Response
from aiohttp import hdrs
from openai import InvalidRequestError, OpenAIError

from aidial_assistant.application.args import parse_args
from aidial_assistant.application.prompts import (
    MAIN_SYSTEM_DIALOG_MESSAGE,
    RESP_DIALOG_PROMPT,
)
from aidial_assistant.application.server_callback import ServerChainCallback
from aidial_assistant.chain.command_chain import CommandChain, CommandDict
from aidial_assistant.chain.model_client import (
    ModelClient,
    ReasonLengthException,
    UsagePublisher,
)
from aidial_assistant.commands.reply import Reply
from aidial_assistant.commands.run_plugin import PluginInfo, RunPlugin
from aidial_assistant.utils.open_ai_plugin import (
    AddonTokenSource,
    get_open_ai_plugin_info,
    get_plugin_auth,
)
from aidial_assistant.utils.state import get_system_prefix, parse_history

logger = logging.getLogger(__name__)


def get_request_args(request: Request) -> dict[str, str]:
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
        raise InvalidRequestError("Missing required addon url.", param="")

    return addon.url


class AssistantApplication(ChatCompletion):
    def __init__(self, config_dir: Path):
        self.args = parse_args(config_dir)

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        chat_args = self.args.openai_conf.dict() | get_request_args(request)

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

        usage_publisher = UsagePublisher()
        command_dict: CommandDict = {
            RunPlugin.token(): lambda: RunPlugin(model, tools, usage_publisher),
            Reply.token(): Reply,
        }
        chain = CommandChain(
            model_client=model,
            name="SERVER",
            resp_prompt=RESP_DIALOG_PROMPT,
            command_dict=command_dict,
        )
        system_message = MAIN_SYSTEM_DIALOG_MESSAGE.render(
            system_prefix=get_system_prefix(request.messages),
            tools=tool_descriptions,
        )
        history = parse_history(
            request.messages,
            system_message,
        )
        choice = response.create_single_choice()
        choice.open()

        callback = ServerChainCallback(choice)
        finish_reason = FinishReason.STOP
        try:
            await chain.run_chat(history, callback, usage_publisher)
        except ReasonLengthException:
            finish_reason = FinishReason.LENGTH
        except OpenAIError as e:
            if e.error:
                raise HTTPException(
                    e.error.message,
                    status_code=e.http_status or 500,
                    code=e.error.code,
                )

            raise

        choice.set_state(callback.state)
        choice.close(finish_reason)

        response.set_usage(
            usage_publisher.prompt_tokens, usage_publisher.completion_tokens
        )
