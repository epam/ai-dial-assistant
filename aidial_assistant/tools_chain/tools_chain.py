import json
from typing import Any

from aidial_sdk.chat_completion import Role

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.history import History
from aidial_assistant.model.model_client import (
    ModelClient,
    Message,
    ExtraResultsCallback,
    ToolCall,
    Tool,
)
from aidial_assistant.tools_chain.tool_runner import ToolRunner


def _publish_command(
    command_callback: CommandCallback, name: str, arguments: str
):
    command_callback.on_command(name)
    args_callback = command_callback.args_callback()
    args_callback.on_args_start()
    arg_callback = args_callback.arg_callback()
    arg_callback.on_arg(arguments)
    arg_callback.on_arg_end()
    args_callback.on_args_end()


class ToolCallsCallback(ExtraResultsCallback):
    def __init__(self):
        self.tool_calls: list[ToolCall] = []

    def on_tool_calls(self, tool_calls: list[ToolCall]):
        self.tool_calls = tool_calls


class ToolsChain:
    def __init__(
        self,
        model: ModelClient,
        tools: list[Tool],
        tool_runner: ToolRunner,
    ):
        self.model = model
        self.tools = tools
        self.tool_runner = tool_runner

    async def run_chat(self, messages: list[Message], callback: ChainCallback):
        result_callback = callback.result_callback()
        while True:
            tool_calls_callback = ToolCallsCallback()
            async for chunk in self.model.agenerate(
                messages, tool_calls_callback, tools=self.tools
            ):
                result_callback.on_result(chunk)

            if not tool_calls_callback.tool_calls:
                break

            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=tool_calls_callback.tool_calls,
                )
            )

            for tool_call in tool_calls_callback.tool_calls:
                function = tool_call["function"]
                name = function["name"]
                arguments = function["arguments"]
                with callback.command_callback() as command_callback:
                    _publish_command(command_callback, name, arguments)
                    try:
                        result = await self.tool_runner.run(
                            name,
                            json.loads(arguments),
                            command_callback.execution_callback(),
                        )
                        messages.append(
                            Message(
                                role=Role.USER,
                                name=name,
                                tool_call_id=tool_call["id"],
                                content=result.text,
                            )
                        )
                        command_callback.on_result(result)
                    except Exception as e:
                        messages.append(
                            Message(
                                role=Role.USER,
                                name=name,
                                tool_call_id=tool_call["id"],
                                content=str(e),
                            )
                        )
                        command_callback.on_error(e)
