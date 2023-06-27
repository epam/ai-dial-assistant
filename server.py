#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
from asyncio import create_task
from pathlib import Path
from typing import Any, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from chains.command_chain import CommandChain
from chains.model_client import ModelClient
from cli.main_args import parse_args
from conf.project_conf import CommandConf, Conf, PluginTool, read_conf
from llm.base import create_chat_from_conf
from main import collect_plugin
from prompts.dialog import RESP_DIALOG_PROMPT, SYSTEM_DIALOG_MESSAGE
from protocol.command_result import Status, responses_to_text, CommandResult
from protocol.commands.base import CommandObject, commands_to_text
from protocol.commands.end_dialog import EndDialog
from protocol.commands.run_plugin import RunPlugin
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import CommandDict, ExecutionContext
from server_callback import ServerChainCallback

app = FastAPI()


def wrap_message(response_id: str, timestamp: float, choice: dict):
    return (
        "data: "
        + json.dumps(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": timestamp,
                "choices": [choice]
            }
        )
        + "\n"
    )


@app.post("/chat/completions")
async def index(request: Request):
    data = await request.json()
    messages = data["messages"]

    commands: dict[str, CommandConf] = {}
    tools: dict[str, PluginTool] = {}

    callback = ServerChainCallback()

    command_dict: CommandDict = {
        RunPlugin.token(): RunPlugin,
        SayOrAsk.token(): EndDialog,
    }

    conf = read_conf(Conf, Path("plugins/index.yaml"))
    for name, plugin in conf.plugins.items():
        collect_plugin(conf.commands, name, plugin, commands, tools, command_dict)

    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)

    history = parse_history(RESP_DIALOG_PROMPT, messages, [
        SYSTEM_DIALOG_MESSAGE.format(commands=commands, tools=tools),
        AIMessage(content=commands_to_text([{"command": SayOrAsk.token(), "args": ["How can I help you?"]}]))
    ])
    response_id = str(uuid.uuid4())
    timestamp = time.time()

    chain = CommandChain(
        model_client=ModelClient(model=model),
        name="SERVER",
        resp_prompt=RESP_DIALOG_PROMPT,
        ctx=ExecutionContext(command_dict),
    )

    async def event_stream():
        producer = create_task(chain.run_chat(history, callback))
        while True:
            item = await callback.queue.get()
            if item is None:
                choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
                message = wrap_message(response_id, timestamp, choice)
                yield message
                yield "data: [DONE]\n"
                break

            choice = {"index": 0, "delta": item}
            message = wrap_message(response_id, timestamp, choice)
            yield message

        await producer

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def parse_command(invocation: str) -> CommandObject:
    args_index = invocation.index("(")
    return {
        "command": invocation[:args_index],
        "args": json.loads(
            "[" + invocation[args_index + 1 : -1].removesuffix(")") + "]"
        ),
    }


def parse_response(response: str) -> CommandResult:
    return {"status": Status.SUCCESS, "response": response}


def sort_by_index(array: List[Any]):
    return array.sort(key=lambda item: int(item["index"]))


def parse_history(
    resp_prompt: HumanMessagePromptTemplate,
    history: List[Any],
    prefix_messages: List[BaseMessage]
) -> List[BaseMessage]:
    init_messages = list(prefix_messages)

    for message in history:
        if message["role"] == "assistant":
            tools = message.get("custom_content", {}).get("state", [])
            sort_by_index(tools)
            for invocation in tools:
                content = invocation["message"]
                if invocation["role"] == "assistant":
                    init_messages.append(AIMessage(content=content))
                else:
                    init_messages.append(resp_prompt.format(responses=content))
            init_messages.append(AIMessage(content=commands_to_text(
                [{"command": SayOrAsk.token(), "args": [message["content"]]}]
            )))

        if message["role"] == "user":
            responses = responses_to_text([parse_response(message["content"])])
            init_messages.append(resp_prompt.format(responses=responses))

    return init_messages


if __name__ == "__main__":
    # Need to add this so that plugin modules residing at "plugins" folder could be references in index.yaml using relative module paths. Otherwise, we need to prefix all module paths with "plugins."
    sys.path.append(os.path.abspath("plugins"))
    uvicorn.run(app, port=5000)
