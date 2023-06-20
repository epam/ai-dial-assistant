#!/usr/bin/env python3
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import Any, List

from flask import Flask, Response, request
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from typing_extensions import override

from chains.command_chain import ChainCallback, CommandChain
from chains.model_client import ModelClient
from cli.main_args import parse_args
from conf.project_conf import CommandConf, Conf, PluginTool, read_conf
from llm.base import create_chat_from_conf
from main import collect_plugin
from prompts.dialog import RESP_DIALOG_PROMPT, SYSTEM_DIALOG_MESSAGE
from protocol.command_result import CommandResultDict, Status, responses_to_text
from protocol.commands.base import CommandObject, commands_to_text
from protocol.commands.run_plugin import RunPlugin
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import CommandDict, ExecutionContext

app = Flask(__name__)


class ChunkCallback(ChainCallback):
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
    return (
        "data: "
        + json.dumps(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": timestamp,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            }
        )
        + "\n"
    )


@app.route("/chat/completions", methods=["POST"])
def index():
    data = request.json
    messages = data["messages"]  # type: ignore

    commands: dict[str, CommandConf] = {}
    tools: dict[str, PluginTool] = {}

    callback = ChunkCallback()

    command_dict: CommandDict = {
        RunPlugin.token(): lambda dict: RunPlugin(callback, dict),
        SayOrAsk.token(): SayOrAsk,
    }

    conf = read_conf(Conf, Path("plugins/index.yaml"))
    for name, plugin in conf.plugins.items():
        collect_plugin(conf.commands, name, plugin, commands, tools, command_dict)

    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)

    history = parse_history(messages, commands, tools)
    response_id = str(uuid.uuid4())
    timestamp = time.time()

    chain = CommandChain(
        model_client=ModelClient(model=model),
        name="SERVER",
        resp_prompt=RESP_DIALOG_PROMPT,
        ctx=ExecutionContext(command_dict),
    )

    def event_stream():
        thread = threading.Thread(target=run_chat, args=(chain, history, callback))
        thread.start()
        while True:
            item = callback.queue.get()
            if item == {}:
                choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
                message = wrap_message(response_id, timestamp, choice)
                yield message
                yield "data: [DONE]\n"
                break

            choice = {"index": 0, "delta": item}
            message = wrap_message(response_id, timestamp, choice)
            yield message

        thread.join()

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Content-Type": "application/json"},
    )


def parse_command(invocation: str) -> CommandObject:
    args_index = invocation.index("(")
    return {
        "command": invocation[:args_index],
        "args": json.loads(
            "[" + invocation[args_index + 1 : -1].removesuffix(")") + "]"
        ),
    }


def parse_response(response: str, id: int) -> CommandResultDict:
    return {"status": Status.SUCCESS, "id": id, "response": response}


def parse_history(
    history: List[Any],
    commands_conf: dict[str, CommandConf],
    tools: dict[str, PluginTool],
) -> List[BaseMessage]:
    init_messages = [SYSTEM_DIALOG_MESSAGE.format(commands=commands_conf, tools=tools)]
    id = 1

    for message in history:
        if message["role"] == "assistant":
            parts = message["content"].split("<<<")
            length = len(parts)
            assert length % 2 == 1
            if length > 1:
                i = 0
                commands: List[CommandObject] = []
                responses: List[CommandResultDict] = []
                while i < length - 1:
                    text = parts[i].strip()
                    if text.startswith(">>>Run command: "):
                        if responses:
                            init_messages.append(
                                HumanMessage(content=responses_to_text(responses))
                            )
                            responses = []

                        command = parse_command(text.removeprefix(">>>Run command: "))
                        commands.append(command)
                    else:
                        if commands:
                            init_messages.append(
                                AIMessage(content=commands_to_text(commands))
                            )
                            commands = []

                        response = parse_response(text.removeprefix(">>>"), id)
                        responses.append(response)
                    i += 1

                if responses:
                    init_messages.append(
                        HumanMessage(content=responses_to_text(responses))
                    )

            command: CommandObject = {
                "command": SayOrAsk.token(),
                "args": [parts[-1].strip()],
            }
            init_messages.append(AIMessage(content=commands_to_text([command])))

        if message["role"] == "user":
            response = parse_response(message["content"], id)
            init_messages.append(HumanMessage(content=responses_to_text([response])))
            id += 1

    return init_messages


if __name__ == "__main__":
    # Need to add this so that plugin modules residing at "plugins" folder could be references in index.yaml using relative module paths. Otherwise, we need to prefix all module paths with "plugins."
    sys.path.append(os.path.abspath("plugins"))
    app.run(port=5000, debug=True)
