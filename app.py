#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
from asyncio import create_task
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from chains.command_chain import CommandChain
from chains.model_client import ModelClient
from cli.main_args import parse_args
from conf.project_conf import PluginOpenAI
from llm.base import create_chat_from_conf
from prompts.dialog import RESP_DIALOG_PROMPT
from protocol.commands.end_dialog import EndDialog
from protocol.commands.run_plugin import RunPlugin
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import CommandDict, ExecutionContext
from server_callback import ServerChainCallback
from utils.open_ai_plugin import get_open_ai_plugin_info
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


def extract_key(authorization: str) -> str:
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix):].strip()

    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.post("/chat/completions")
async def index(request: Request):
    data = await request.json()
    messages = data["messages"]
    addons = data.get("addons", [])

    tools: dict[str, PluginOpenAI] = {}
    plugin_descriptions: dict[str, str] = {}
    for addon in addons:
        info = get_open_ai_plugin_info(addon["url"])
        tools[info.ai_plugin.name_for_model] = PluginOpenAI(type="open-ai-plugin", url=addon["url"])
        plugin_descriptions[info.ai_plugin.name_for_model] = or_else(
            info.open_api.info.description, info.ai_plugin.description_for_human)

    args = parse_args()
    openai_api_key = extract_key(request.headers.get("Authorization", ""))
    model = create_chat_from_conf(args.openai_conf, args.chat_conf, openai_api_key)

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
    uvicorn.run(app, port=8080)
