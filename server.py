#!/usr/bin/env python3
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import List, Any

from flask import Flask, Response, request
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from typing_extensions import override

from chains.command_chain import CommandChain, ChainCallback
from chains.model_client import ModelClient
from cli.main_args import parse_args
from conf.project_conf import CommandConf, PluginTool, read_conf, Conf
from llm.base import create_chat_from_conf
from main import collect_plugin
from prompts.dialog import SYSTEM_DIALOG_MESSAGE, RESP_DIALOG_PROMPT
from protocol.commands.run_plugin import RunPlugin
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import CommandDict, ExecutionContext

app = Flask(__name__)


class MyCallback(ChainCallback):
    queue = Queue[Any]()

    @override
    def on_message(self, role: str | None = None, text: str | None = None):
        item = {}
        if role is not None:
            item["role"] = role
        if text is not None:
            item["content"] = text

        self.queue.put(item)

    @override
    def on_end(self, error: str | None = None):
        self.queue.put({})


def run_chat(chain: CommandChain, messages: List[BaseMessage], callback: ChainCallback):
    chain.run_chat(messages, callback)


def wrap_message(response_id: str, timestamp: float, choice: dict):
    return "data: " + json.dumps({
        "id": response_id,
        "object": "chat.completion",
        "created": timestamp,
        "choices": [choice],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }) + "\n"


@app.route('/chat/completions', methods=['POST'])
def index():
    data = request.json
    messages = data['messages']

    commands: dict[str, CommandConf] = {}
    tools: dict[str, PluginTool] = {}

    command_dict: CommandDict = {
        RunPlugin.token(): RunPlugin,
        SayOrAsk.token(): SayOrAsk,
    }

    conf = read_conf(Conf, Path("plugins/index.yaml"))
    for name, plugin in conf.plugins.items():
        collect_plugin(conf.commands, name, plugin, commands, tools, command_dict)

    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)

    chain = CommandChain(
        model_client=ModelClient(model=model),
        name="server",
        resp_prompt=RESP_DIALOG_PROMPT,
        ctx=ExecutionContext(command_dict),
    )

    history = parse_history(messages, commands, tools)
    callback = MyCallback()
    response_id = str(uuid.uuid4())
    timestamp = time.time()

    def event_stream():
        thread = threading.Thread(target=run_chat, args=(chain, history, callback))
        thread.start()
        while True:
            item = callback.queue.get()
            if item == {}:
                choice = {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
                message = wrap_message(response_id, timestamp, choice)
                # print(message, end="")
                yield message
                yield "data: [DONE]\n"
                break

            choice = {
                "index": 0,
                "delta": item
            }
            message = wrap_message(response_id, timestamp, choice)
            # print(message, end="")
            yield message

        thread.join()

    return Response(event_stream(), mimetype="text/event-stream", headers={'Content-Type': 'application/json'})


def parse_command(invocation: str):
    args_index = invocation.index("(")
    return {
        "command": invocation[:args_index],
        "args": json.loads("[" + invocation[args_index + 1:-1].removesuffix(")") + "]")
    }


def parse_response(response: str, response_id: int):
    return {
        "status": "SUCCESS",
        "response_id": response_id,
        "response": response
    }


def parse_history(
        history: List[Any],
        commands: dict[str, CommandConf],
        tools: dict[str, PluginTool]) -> List[BaseMessage]:
    init_messages = [
        SYSTEM_DIALOG_MESSAGE.format(commands=commands, tools=tools),
        AIMessage(content=json.dumps({"commands": [{"command": SayOrAsk.token(), "args": ["How can I help you?"]}]}))
    ]
    response_id = 1
    for message in history:
        if message["role"] == "assistant":
            parts = message["content"].split("<<<")
            length = len(parts)
            assert length % 2 == 1
            if length > 1:
                i = 0
                commands = []
                responses = []
                while i < length - 1:
                    text = parts[i].strip()
                    if text.startswith(">>>Run command: "):
                        if responses:
                            init_messages.append(HumanMessage(content=json.dumps({"responses": responses})))
                            responses = []

                        command = parse_command(text.removeprefix(">>>Run command: "))
                        commands.append(command)
                    else:
                        if commands:
                            init_messages.append(AIMessage(content=json.dumps({"commands": commands})))
                            commands = []

                        response = parse_response(text.removeprefix(">>>"), response_id)
                        responses.append(response)
                    i += 1

                if responses:
                    init_messages.append(HumanMessage(content=json.dumps({"responses": responses})))

            command = {
                "command": "say-or-ask",
                "args": [parts[-1].strip()]
            }
            init_messages.append(AIMessage(content=json.dumps({"commands": [command]})))

        if message["role"] == "user":
            response = parse_response(message["content"], response_id)
            init_messages.append(HumanMessage(content=json.dumps({"responses": [response]})))
            response_id += 1

    return init_messages


if __name__ == "__main__":
    # Need to add this so that plugin modules residing at "plugins" folder could be references in index.yaml using relative module paths. Otherwise, we need to prefix all module paths with "plugins."
    sys.path.append(os.path.abspath("plugins"))
    app.run(port=5000, debug=True)
