import json
from typing import Any

from aidial_sdk.chat_completion import Role
from openai import BadRequestError

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.command_chain import CommandDict
from aidial_assistant.chain.command_result import (
    CommandInvocation,
    CommandResult,
    Status,
    commands_to_text,
    responses_to_text,
)
from aidial_assistant.chain.model_response_reader import (
    AssistantProtocolException,
)
from aidial_assistant.commands.base import Command
from aidial_assistant.model.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    ToolCall,
)
from aidial_assistant.utils.open_ai import Tool


def _publish_command(
    command_callback: CommandCallback, name: str, arguments: str
):
    command_callback.on_command(name)
    args_callback = command_callback.args_callback()
    args_callback.on_args_start()
    args_callback.on_args_chunk(arguments)
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
        command_dict: CommandDict,
    ):
        self.model = model
        self.tools = tools
        self.command_dict = command_dict

    async def run_chat(self, messages: list[Message], callback: ChainCallback):
        result_callback = callback.result_callback()
        dialogue: list[Message] = []
        last_message_message_count = 0
        while True:
            tool_calls_callback = ToolCallsCallback()
            try:
                async for chunk in self.model.agenerate(
                    messages + dialogue, tool_calls_callback, tools=self.tools
                ):
                    result_callback.on_result(chunk)
            except BadRequestError as e:
                if len(dialogue) == 0 or e.code == "429":
                    raise

                dialogue = dialogue[:-last_message_message_count]
                async for chunk in self.model.agenerate(
                    messages + dialogue, tool_calls_callback
                ):
                    result_callback.on_result(chunk)
                break

            if not tool_calls_callback.tool_calls:
                break

            result_messages = await self._process_tools(
                tool_calls_callback.tool_calls, callback
            )
            dialogue.append(
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=tool_calls_callback.tool_calls,
                )
            )
            dialogue.extend(result_messages)
            last_message_message_count = len(result_messages) + 1

    def _create_command(self, name: str) -> Command:
        if name not in self.command_dict:
            raise AssistantProtocolException(
                f"The tool '{name}' is expected to be one of {list(self.command_dict.keys())}"
            )

        return self.command_dict[name]()

    async def _process_tools(
        self, tool_calls: list[ToolCall], callback: ChainCallback
    ):
        commands: list[CommandInvocation] = []
        command_results: list[CommandResult] = []
        result_messages: list[Message] = []
        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            arguments: dict[str, Any] = json.loads(function["arguments"])
            with callback.command_callback() as command_callback:
                _publish_command(command_callback, name, json.dumps(arguments))
                command = self._create_command(name)
                result = await self._execute_command(
                    command,
                    arguments,
                    command_callback,
                )
                result_messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=tool_call["id"],
                        content=result["response"],
                    )
                )
                command_results.append(result)

            commands.append(
                CommandInvocation(command=name, arguments=arguments)
            )

        callback.on_state(
            commands_to_text(commands), responses_to_text(command_results)
        )

        return result_messages

    @staticmethod
    async def _execute_command(
        command: Command,
        args: dict[str, Any],
        command_callback: CommandCallback,
    ) -> CommandResult:
        try:
            result = await command.execute(
                args, command_callback.execution_callback()
            )
            command_callback.on_result(result)
            return CommandResult(status=Status.SUCCESS, response=result.text)
        except Exception as e:
            command_callback.on_error(e)
            return CommandResult(status=Status.ERROR, response=str(e))
