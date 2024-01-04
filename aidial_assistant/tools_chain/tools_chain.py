import json
from typing import Any

from aidial_sdk.chat_completion import Role

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.command_result import (
    CommandInvocation,
    CommandResult,
    Status,
    commands_to_text,
    responses_to_text,
)
from aidial_assistant.model.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    Tool,
    ToolCall,
)
from aidial_assistant.tools_chain.tool_runner import ToolRunner


def _publish_command(
    command_callback: CommandCallback, name: str, arguments: dict[str, Any]
):
    command_callback.on_command(name)
    args_callback = command_callback.args_callback()
    args_callback.on_args(arguments)


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

            commands: list[CommandInvocation] = []
            results: list[CommandResult] = []
            for tool_call in tool_calls_callback.tool_calls:
                function = tool_call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
                commands.append(CommandInvocation(command=name, args=arguments))
                with callback.command_callback() as command_callback:
                    _publish_command(command_callback, name, arguments)
                    try:
                        result = await self.tool_runner.run(
                            name,
                            arguments,
                            command_callback.execution_callback(),
                        )
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                tool_call_id=tool_call["id"],
                                content=result.text,
                            )
                        )
                        command_callback.on_result(result)
                        results.append(
                            CommandResult(
                                status=Status.SUCCESS, response=result.text
                            )
                        )
                    except Exception as e:
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                tool_call_id=tool_call["id"],
                                content=str(e),
                            )
                        )
                        command_callback.on_error(e)
                        results.append(
                            CommandResult(status=Status.ERROR, response=str(e))
                        )

            callback.on_state(
                commands_to_text(commands), responses_to_text(results)
            )
