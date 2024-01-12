import json
from typing import Any, Tuple, cast

from openai import BadRequestError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.command_chain import CommandConstructor
from aidial_assistant.chain.command_result import (
    CommandInvocation,
    CommandResult,
    Commands,
    Responses,
    Status,
    commands_to_text,
    responses_to_text,
)
from aidial_assistant.chain.history import MessageScope, ScopedMessage
from aidial_assistant.chain.model_response_reader import (
    AssistantProtocolException,
)
from aidial_assistant.commands.base import Command
from aidial_assistant.model.model_client import (
    ExtraResultsCallback,
    ModelClient,
)
from aidial_assistant.utils.exceptions import RequestParameterValidationError
from aidial_assistant.utils.open_ai import tool_calls_message, tool_message


def convert_commands_to_tools(
    scoped_messages: list[ScopedMessage],
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    next_tool_id: int = 0
    last_call_count: int = 0
    for scoped_message in scoped_messages:
        message = scoped_message.message
        if scoped_message.scope == MessageScope.INTERNAL:
            content = cast(str, message.get("content"))
            if not content:
                raise RequestParameterValidationError(
                    "State is broken. Content cannot be empty.",
                    param="messages",
                )

            if message["role"] == "assistant":
                commands: Commands = json.loads(content)
                messages.append(
                    tool_calls_message(
                        [
                            ChatCompletionMessageToolCallParam(
                                id=str(next_tool_id + index),
                                function=Function(
                                    name=command["command"],
                                    arguments=json.dumps(command["arguments"]),
                                ),
                                type="function",
                            )
                            for index, command in enumerate(
                                commands["commands"]
                            )
                        ],
                    )
                )
                last_call_count = len(commands["commands"])
                next_tool_id += last_call_count
            elif message["role"] == "user":
                responses: Responses = json.loads(content)
                response_count = len(responses["responses"])
                if response_count != last_call_count:
                    raise RequestParameterValidationError(
                        f"Expected {last_call_count} responses, but got {response_count}.",
                        param="messages",
                    )
                first_tool_id = next_tool_id - last_call_count
                messages.extend(
                    [
                        tool_message(
                            content=response["response"],
                            tool_call_id=str(first_tool_id + index),
                        )
                        for index, response in enumerate(responses["responses"])
                    ]
                )
        else:
            messages.append(scoped_message.message)
    return messages


def _publish_command(
    command_callback: CommandCallback, name: str, arguments: str
):
    command_callback.on_command(name)
    args_callback = command_callback.args_callback()
    args_callback.on_args_start()
    args_callback.on_args_chunk(arguments)
    args_callback.on_args_end()


CommandTool = Tuple[CommandConstructor, ChatCompletionToolParam]
CommandToolDict = dict[str, CommandTool]


class ToolCallsCallback(ExtraResultsCallback):
    def __init__(self):
        self.tool_calls: list[ChatCompletionMessageToolCallParam] = []

    def on_tool_calls(
        self, tool_calls: list[ChatCompletionMessageToolCallParam]
    ):
        self.tool_calls = tool_calls


class ToolsChain:
    def __init__(self, model: ModelClient, command_tool_dict: CommandToolDict):
        self.model = model
        self.command_tool_dict = command_tool_dict

    async def run_chat(
        self,
        messages: list[ChatCompletionMessageParam],
        callback: ChainCallback,
    ):
        result_callback = callback.result_callback()
        dialogue: list[ChatCompletionMessageParam] = []
        last_message_block_length = 0
        tools = [tool for _, tool in self.command_tool_dict.values()]
        while True:
            tool_calls_callback = ToolCallsCallback()
            try:
                async for chunk in self.model.agenerate(
                    messages + dialogue, tool_calls_callback, tools=tools
                ):
                    result_callback.on_result(chunk)
            except BadRequestError as e:
                if len(dialogue) == 0 or e.code == "429":
                    raise

                # If the dialog size exceeds model context size then remove last message block
                # and try again without tools.
                dialogue = dialogue[:-last_message_block_length]
                async for chunk in self.model.agenerate(
                    messages + dialogue, tool_calls_callback
                ):
                    result_callback.on_result(chunk)
                break

            if not tool_calls_callback.tool_calls:
                break

            dialogue.append(
                tool_calls_message(
                    tool_calls_callback.tool_calls,
                )
            )
            result_messages = await self._run_tools(
                tool_calls_callback.tool_calls, callback
            )
            dialogue.extend(result_messages)
            last_message_block_length = len(result_messages) + 1

    def _create_command(self, name: str) -> Command:
        if name not in self.command_tool_dict:
            raise AssistantProtocolException(
                f"The tool '{name}' is expected to be one of {list(self.command_tool_dict.keys())}"
            )

        command, _ = self.command_tool_dict[name]

        return command()

    async def _run_tools(
        self,
        tool_calls: list[ChatCompletionMessageToolCallParam],
        callback: ChainCallback,
    ):
        commands: list[CommandInvocation] = []
        command_results: list[CommandResult] = []
        result_messages: list[ChatCompletionMessageParam] = []
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
                    tool_message(
                        content=result["response"],
                        tool_call_id=tool_call["id"],
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
