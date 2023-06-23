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

from chains.command_chain import ChainCallback, CommandChain, CommandCallback, ArgsCallback, ArgCallback, \
    ExecutionCallback, ResultCallback
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

app = Flask(__name__)


class ServerArgCallback(ArgCallback):
    def __init__(self, command_index: int, arg_index: int, queue: Queue[Any]):
        self.command_index = command_index
        self.arg_index = arg_index
        self.queue = queue

    def on_arg_start(self):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": '"' if self.arg_index == 0 else ', "'}]}})

    def on_arg(self, token: str):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": token.replace('"', '\\"')}]}})

    def on_arg_end(self):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": '"'}]}})


class ServerArgsCallback(ArgsCallback):
    def __init__(self, command_index: int, queue: Queue[Any]):
        self.command_index = command_index
        self.queue = queue
        self.arg_index = -1

    def on_args_start(self):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": "("}]}})

    def arg_callback(self) -> ArgCallback:
        self.arg_index += 1
        return ServerArgCallback(self.command_index, self.arg_index, self.queue)

    def on_args_end(self):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": ")\n"}]}})


class ServerExecutionCallback(ExecutionCallback):
    def __init__(self, command_index: int, queue: Queue[Any]):
        self.command_index = command_index
        self.queue = queue

    @override
    def on_message(self, token: str):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": token}]}})


class ServerCommandCallback(CommandCallback):
    def __init__(self, command_index: int, queue: Queue[Any]):
        self.command_index = command_index
        self.queue = queue

    @override
    def on_command(self, command: str):
        self.queue.put(
            {"custom_content": {"stages": [{"index": self.command_index, "content": "Running command: " + command}]}})

    @override
    def execution_callback(self) -> ServerExecutionCallback:
        return ServerExecutionCallback(self.command_index, self.queue)

    @override
    def args_callback(self) -> ArgsCallback:
        return ServerArgsCallback(self.command_index, self.queue)

    @override
    def on_result(self, response):
        self.queue.put({"custom_content": {"stages": [{"index": self.command_index, "content": f"Result: {response}"}]}})


class ServerResultCallback(ResultCallback):
    def __init__(self, queue: Queue[Any]):
        self.queue = queue

    def on_start(self):
        pass

    def on_result(self, token):
        self.queue.put({"content": token})

    def on_end(self):
        pass


class ServerChainCallback(ChainCallback):
    def __init__(self):
        self.command_index: int = -1
        self.message_index: int = -1
        self.queue = Queue[Any]()

    @override
    def on_start(self):
        self.queue.put({"role": "assistant"})

    @override
    def command_callback(self) -> CommandCallback:
        self.command_index += 1
        return ServerCommandCallback(self.command_index, self.queue)

    @override
    def on_ai_message(self, message: str):
        self.message_index += 1
        self.queue.put(
            {"custom_content": {"state": [{"index": self.message_index, "role": "assistant", "message": message}]}})

    @override
    def on_human_message(self, message: str):
        self.message_index += 1
        self.queue.put(
            {"custom_content": {"state": [{"index": self.message_index, "role": "user", "message": message}]}})

    @override
    def result_callback(self) -> ResultCallback:
        return ServerResultCallback(self.queue)

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
                "choices": [choice]
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

    history = parse_history(messages, [
        SYSTEM_DIALOG_MESSAGE.format(commands=commands, tools=tools),
        AIMessage(content=commands_to_text([{"command": SayOrAsk.token(), "args": ["How can I help you?"]}])),
    ])
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


def parse_response(response: str) -> CommandResult:
    return {"status": Status.SUCCESS, "response": response}


def sort_by_index(array: List[Any]):
    return array.sort(key=lambda item: int(item["index"]))


def parse_history(
    history: List[Any],
    prefix_messages: List[BaseMessage]
) -> List[BaseMessage]:
    init_messages = list(prefix_messages)

    for message in history:
        if message["role"] == "assistant":
            tools = message.get("custom_content", {}).get("state", [])
            sort_by_index(tools)
            if tools:
                for invocation in tools:
                    sort_by_index(invocation["invocation"])
                    commands = list[CommandObject]()
                    responses = list[CommandResult]()
                    for command in invocation["invocation"]:
                        commands.append(
                            {"command": command["command"], "args": command["args"]})
                        response = command["response"]
                        responses.append(
                            {"status": response["status"], "response": response["content"]})

                    init_messages.append(AIMessage(content=commands_to_text(commands)))
                    init_messages.append(HumanMessage(content=responses_to_text(responses)))
            init_messages.append(AIMessage(content=commands_to_text(
                [{"command": SayOrAsk.token(), "args": [message["content"]]}]
            )))

        if message["role"] == "user":
            response = parse_response(message["content"])
            init_messages.append(HumanMessage(content=responses_to_text([response])))

    return init_messages


if __name__ == "__main__":
    # Need to add this so that plugin modules residing at "plugins" folder could be references in index.yaml using relative module paths. Otherwise, we need to prefix all module paths with "plugins."
    sys.path.append(os.path.abspath("plugins"))
    app.run(port=5000, debug=True)
