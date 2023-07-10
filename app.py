#!/usr/bin/env python3
import json
import time
import uuid
from asyncio import create_task
from typing import Any

import uvicorn
from aiohttp import hdrs
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from langchain.chat_models import ChatOpenAI
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.status import HTTP_401_UNAUTHORIZED

from chains.command_chain import CommandChain
from chains.model_client import ModelClient
from cli.main_args import parse_args
from llm.base import create_azure_chat
from prompts.dialog import RESP_DIALOG_PROMPT
from protocol.commands.end_dialog import EndDialog
from protocol.commands.run_plugin import RunPlugin
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import CommandDict, ExecutionContext
from server_callback import ServerChainCallback
from utils.addon_token_source import AddonTokenSource
from utils.open_ai_plugin import get_open_ai_plugin_info, OpenAIPluginInfo
from utils.optional import or_else
from utils.state import parse_history

app = FastAPI()


def create_chunk(response_id: str, timestamp: float, choice: dict[str, Any]):
    return (
        "data: "
        + json.dumps(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": timestamp,
                "choices": [{"index": 0} | choice]
            }
        )
        + "\n"
    )


def get_request_args(payload: dict, api_version: str | None, user_auth: str | None) -> dict[str, str]:
    args = {
        "model_name": payload.get("model"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "stop": payload.get("stop"),
        "openai_api_version": api_version,
        "user": payload.get("user"),
        "headers": None if user_auth is None else {hdrs.AUTHORIZATION: user_auth}
    }

    return {k: v for k, v in args.items() if v is not None}


@app.post("/openai/deployments/{service_name}/chat/completions")
async def azure(service_name: str, request: Request) -> Response:
    if service_name != "assistant":
        raise HTTPException(status_code=404)

    args = parse_args()
    data = await request.json()
    user_auth = request.headers.get(hdrs.AUTHORIZATION)
    chat_args = args.openai_conf.dict() | get_request_args(data, request.query_params.get("api-version"), user_auth)

    model = create_azure_chat(chat_args, request.headers["api-key"])

    addons = [addon["url"] for addon in data.get("addons", [])]
    token_source = AddonTokenSource(request.headers, addons)
    return await process_request(model, data["messages"], addons, token_source)


@app.get("/healthcheck/status200")
def status200() -> Response:
    return Response("Service is running...", status_code=200)


async def process_request(
        model: ChatOpenAI, messages: list[Any], addons: list[str], token_source: AddonTokenSource) -> Response:
    tools: dict[str, OpenAIPluginInfo] = {}
    plugin_descriptions: dict[str, str] = {}
    for addon in addons:
        info = await get_open_ai_plugin_info(addon, token_source)
        tools[info.ai_plugin.name_for_model] = info
        plugin_descriptions[info.ai_plugin.name_for_model] = or_else(
            info.open_api.info.description, info.ai_plugin.description_for_human)

    command_dict: CommandDict = {
        RunPlugin.token(): lambda: RunPlugin(model, tools),
        SayOrAsk.token(): EndDialog,
    }

    history = parse_history(messages, plugin_descriptions)
    response_id = str(uuid.uuid4())
    timestamp = time.time()

    chain = CommandChain(
        model_client=ModelClient(model=model),
        name="SERVER",
        resp_prompt=RESP_DIALOG_PROMPT,
        ctx=ExecutionContext(command_dict),
    )

    async def event_stream():
        callback = ServerChainCallback()
        producer = create_task(chain.run_chat(history, callback))
        while True:
            item = await callback.queue.get()
            if item is None:
                yield create_chunk(response_id, timestamp, {"delta": {}, "finish_reason": "stop"})
                yield "data: [DONE]\n"
                break

            yield create_chunk(response_id, timestamp, {"delta": item})

        await producer

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, port=7001)
