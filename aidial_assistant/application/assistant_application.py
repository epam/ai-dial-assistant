from pathlib import Path

from aidial_sdk.chat_completion.base import ChatCompletion
from aidial_sdk.chat_completion.request import Addon, Request
from aidial_sdk.chat_completion.response import Response
from aiohttp import hdrs
from openai import InvalidRequestError

from aidial_assistant.application.args import parse_args
from aidial_assistant.application.prompts import RESP_DIALOG_PROMPT
from aidial_assistant.application.server_callback import ServerChainCallback
from aidial_assistant.chain.command_chain import CommandChain
from aidial_assistant.chain.execution_context import (
    CommandDict,
    ExecutionContext,
)
from aidial_assistant.chain.model_client import ModelClient
from aidial_assistant.commands.reply import Reply
from aidial_assistant.commands.run_plugin import RunPlugin
from aidial_assistant.utils.open_ai_plugin import (
    AddonTokenSource,
    OpenAIPluginInfo,
    get_open_ai_plugin_info,
)
from aidial_assistant.utils.state import parse_history


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

        tools: dict[str, OpenAIPluginInfo] = {}
        plugin_descriptions: dict[str, str] = {}
        for addon in addons:
            info = await get_open_ai_plugin_info(addon, token_source)
            tools[info.ai_plugin.name_for_model] = info
            plugin_descriptions[info.ai_plugin.name_for_model] = (
                info.open_api.info.description
                or info.ai_plugin.description_for_human
            )

        command_dict: CommandDict = {
            RunPlugin.token(): lambda: RunPlugin(model, tools),
            Reply.token(): Reply,
        }

        history = parse_history(
            request.messages,
            plugin_descriptions,
        )
        chain = CommandChain(
            model_client=model,
            name="SERVER",
            resp_prompt=RESP_DIALOG_PROMPT,
            ctx=ExecutionContext(command_dict),
        )
        with response.create_single_choice() as choice:
            callback = ServerChainCallback(choice)
            await chain.run_chat(history, callback)
            choice.set_state(callback.state)
