#!/usr/bin/env python3
import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from aiohttp import hdrs
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.responses import Response, FileResponse, JSONResponse

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
from utils.open_ai import merge, wrap_choice, wrap_error, wrap_chunk
from utils.open_ai_plugin import get_open_ai_plugin_info, OpenAIPluginInfo
from utils.optional import or_else
from utils.state import parse_history

app = FastAPI()


def get_request_args(payload: dict, api_version: str | None, user_auth: str | None) -> dict[str, str]:
    args = {
        "model_name": payload.get("model"),
        "temperature": payload.get("temperature"),
        # "max_tokens": payload.get("max_tokens"), ignore tokens for now, it's tricky to calculate
        "stop": payload.get("stop"),
        "openai_api_version": api_version,
        "user": payload.get("user"),
        "headers": None if user_auth is None else {hdrs.AUTHORIZATION: user_auth}
    }

    return {k: v for k, v in args.items() if v is not None}


@app.post("/openai/deployments/assistant/chat/completions")
async def assistant(request: Request) -> Response:
    args = parse_args()
    data = await request.json()
    user_auth = request.headers.get(hdrs.AUTHORIZATION)
    chat_args = args.openai_conf.dict() | get_request_args(data, request.query_params.get("api-version"), user_auth)

    model = ModelClient(
        model=create_azure_chat(chat_args, request.headers["api-key"]),
        buffer_size=args.chat_conf.buffer_size)

    addons = [addon["url"] for addon in data.get("addons", [])]
    token_source = AddonTokenSource(request.headers, addons)
    response_builder = stream_response if data.get("stream") else plain_response
    return await response_builder(process_request(model, data["messages"], addons, token_source))


@app.get("/healthcheck/status200")
def status200() -> Response:
    return Response("Service is running...", status_code=200)


async def process_request(
        model_client: ModelClient,
        messages: list[Any],
        addons: list[str],
        token_source: AddonTokenSource) -> AsyncIterator[Any]:
    tools: dict[str, OpenAIPluginInfo] = {}
    plugin_descriptions: dict[str, str] = {}
    for addon in addons:
        info = await get_open_ai_plugin_info(addon, token_source)
        tools[info.ai_plugin.name_for_model] = info
        plugin_descriptions[info.ai_plugin.name_for_model] = or_else(
            info.open_api.info.description, info.ai_plugin.description_for_human)

    command_dict: CommandDict = {
        RunPlugin.token(): lambda: RunPlugin(model_client, tools),
        Reply.token(): Reply,
    }

    history = parse_history(messages, plugin_descriptions)
    chain = CommandChain(
        model_client=model_client,
        name="SERVER",
        resp_prompt=RESP_DIALOG_PROMPT,
        ctx=ExecutionContext(command_dict),
    )

    callback = ServerChainCallback()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(chain.run_chat(history, callback))
            while True:
                item = await callback.queue.get()
                if item is None:
                    break

                yield item
    except ExceptionGroup as e:
        raise e.exceptions[0]


async def stream_response(chunks: AsyncIterator[Any]) -> Response:
    async def event_stream():
        response_id = str(uuid.uuid4())
        timestamp = int(time.time())

        try:
            async for chunk in chunks:
                yield wrap_choice(response_id, timestamp, {"delta": chunk})

            yield wrap_choice(response_id, timestamp, {"delta": {}, "finish_reason": "stop"})
        except Exception as e:
            yield wrap_error(e)
        finally:
            yield wrap_chunk("[DONE]")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def plain_response(chunks: AsyncIterator[Any]) -> Response:
    message = {}
    async for chunk in chunks:
        message = merge(message, chunk)

    return JSONResponse({
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {}  # required by langchain
    })


@app.get("/{plugin}/.well-known/{filename}")
def read_file(plugin: str, filename: str):
    return FileResponse(f"{plugin}/.well-known/{filename}")


if __name__ == "__main__":
    uvicorn.run(app, port=7001)
