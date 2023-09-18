from aidial_sdk import ChatCompletion, ChatCompletionRequest, ChatCompletionResponse
from aidial_sdk.chat_completion.request import Addon
from aiohttp import hdrs
from openai import InvalidRequestError
from starlette.datastructures import Headers

from chains.command_chain import CommandChain
from chains.model_client import ModelClient
from cli.main_args import parse_args
from llm.base import create_azure_chat
from prompts.dialog import RESP_DIALOG_PROMPT
from protocol.commands.end_dialog import Reply
from protocol.commands.run_plugin import RunPlugin
from protocol.execution_context import CommandDict, ExecutionContext
from server_callback import ServerChainCallback
from utils.addon_token_source import AddonTokenSource
from utils.open_ai_plugin import OpenAIPluginInfo, get_open_ai_plugin_info
from utils.optional import or_else
from utils.state import parse_history


def get_request_args(request: ChatCompletionRequest) -> dict[str, str]:
    args = {
        "model_name": request.model,
        "temperature": request.temperature,
        # "max_tokens": payload.get("max_tokens"), ignore tokens for now, it's tricky to calculate
        "stop": request.stop,
        # "openai_api_version": request.api_version, -- after the api_version is added to the SDK
        "openai_api_version": "2023-03-15-preview",
        "user": request.user,
        "headers": None if request.jwt is None else {hdrs.AUTHORIZATION: request.jwt},
    }

    return {k: v for k, v in args.items() if v is not None}


def _extract_addon_url(addon: Addon) -> str:
    if addon.url is None:
        raise InvalidRequestError("Missing required addon url.", param="")

    return addon.url


class AssistantApplication(ChatCompletion):
    async def chat_completion(
        self, request: ChatCompletionRequest, response: ChatCompletionResponse
    ) -> None:
        args = parse_args()
        chat_args = args.openai_conf.dict() | get_request_args(request)

        model = ModelClient(
            model=create_azure_chat(chat_args, request.api_key),
            buffer_size=args.chat_conf.buffer_size,
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
            plugin_descriptions[info.ai_plugin.name_for_model] = or_else(
                info.open_api.info.description, info.ai_plugin.description_for_human
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
