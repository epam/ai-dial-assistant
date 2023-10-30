import json
import logging
from typing import Any, AsyncIterator, Callable, List

from aidial_sdk.chat_completion.request import Message, Role
from jinja2 import Template

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.chain.command_result import (
    CommandResult,
    Status,
    responses_to_text,
)
from aidial_assistant.chain.model_client import ModelClient, UsagePublisher
from aidial_assistant.chain.model_response_reader import (
    AssistantProtocolException,
    CommandsReader,
)
from aidial_assistant.commands.base import Command, FinalCommand
from aidial_assistant.json_stream.json_node import (
    JsonNode,
    JsonParsingException,
)
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_parser import JsonParser
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.json_stream.tokenator import AsyncPeekable, Tokenator

logger = logging.getLogger(__name__)

MAX_MESSAGE_COUNT = 20
MAX_RETRY_COUNT = 2

CommandConstructor = Callable[[], Command]
CommandDict = dict[str, CommandConstructor]


class BufferedStream(AsyncIterator[str]):
    def __init__(self, stream: AsyncIterator[str]):
        self.stream = stream
        self.buffer = ""

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        chunk = await anext(self.stream)
        self.buffer += chunk
        return chunk


class CommandChain:
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        command_dict: CommandDict,
        resp_prompt: Template,
    ):
        self.name = name
        self.model_client = model_client
        self.command_dict = command_dict
        self.resp_prompt = resp_prompt

    def _log_message(self, role: Role, content: str | None):
        logger.debug(f"[{self.name}] {role.value}: {content or ''}")

    async def run_chat(
        self,
        history: List[Message],
        callback: ChainCallback,
        usage_publisher: UsagePublisher,
    ) -> str:
        for message in history[:-1]:
            self._log_message(message.role, message.content)

        retry_count = 0
        for _ in range(MAX_MESSAGE_COUNT):
            token_stream = BufferedStream(
                self.model_client.agenerate(
                    self._reinforce_last_message(history), usage_publisher
                )
            )
            tokenator = Tokenator(token_stream)
            await CommandChain._skip_text(tokenator)
            try:
                commands: list[dict[str, Any]] = []
                responses: list[CommandResult] = []
                async with JsonParser.parse(tokenator) as root_node:
                    request_reader = CommandsReader(root_node)
                    async for invocation in request_reader.parse_invocations():
                        command_name = await invocation.parse_name()
                        command = self._create_command(command_name)
                        args = invocation.parse_args()
                        if isinstance(command, FinalCommand):
                            if len(responses) > 0:
                                continue
                            arg = await anext(args)
                            result = await CommandChain._to_result(
                                arg
                                if isinstance(arg, JsonString)
                                else arg.to_string_tokens(),
                                # Some relatively large number to avoid CxSAST warning about potential DoS attack.
                                # Later, the upper limit will be provided by the DIAL Core (proxy).
                                32000,
                                callback.result_callback(),
                            )
                            self._log_message(
                                Role.ASSISTANT, json.dumps(root_node.value())
                            )
                            return result
                        else:
                            response = await CommandChain._execute_command(
                                command_name, command, args, callback
                            )

                            commands.append(invocation.node.value())
                            responses.append(response)

                    if len(responses) == 0:
                        # Assume the model has nothing to say
                        self._log_message(
                            Role.ASSISTANT, json.dumps(root_node.value())
                        )
                        return ""

                normalized_model_response = json.dumps({"commands": commands})
                history.append(
                    Message(
                        role=Role.ASSISTANT, content=normalized_model_response
                    )
                )

                response_text = responses_to_text(responses)
                history.append(Message(role=Role.USER, content=response_text))

                callback.on_state(normalized_model_response, response_text)
                retry_count = 0
            except (JsonParsingException, AssistantProtocolException) as e:
                logger.exception("Failed to process model response")
                callback.on_error(
                    "Error"
                    if retry_count == 0
                    else f"Error (retry {retry_count})",
                    e,
                )

                retry_count += 1
                if retry_count > MAX_RETRY_COUNT:
                    raise e

                if token_stream.buffer:
                    history.append(
                        Message(
                            role=Role.ASSISTANT, content=token_stream.buffer
                        )
                    )
                    history.append(
                        Message(
                            role=Role.USER,
                            content=json.dumps({"error": str(e)}),
                        )
                    )
            finally:
                self._log_message(Role.ASSISTANT, token_stream.buffer)

        raise Exception(f"Max message count of {MAX_MESSAGE_COUNT} exceeded")

    def _reinforce_last_message(self, history: list[Message]) -> list[Message]:
        reinforced_message = self.resp_prompt.render(
            response=history[-1].content
        )
        self._log_message(Role.USER, reinforced_message)
        return history[:-1] + [
            Message(role=Role.USER, content=reinforced_message),
        ]

    def _create_command(self, name: str) -> Command:
        if name not in self.command_dict:
            raise Exception(
                f"The command '{name}' is expected to be one of {[*self.command_dict.keys()]}"
            )

        return self.command_dict[name]()

    @staticmethod
    async def _to_args(
        args: AsyncIterator[JsonNode], callback: CommandCallback
    ) -> AsyncIterator[Any]:
        args_callback = callback.args_callback()
        args_callback.on_args_start()
        async for arg in args:
            arg_callback = args_callback.arg_callback()
            arg_callback.on_arg_start()
            result = ""
            async for token in arg.to_string_tokens():
                arg_callback.on_arg(token)
                result += token
            arg_callback.on_arg_end()
            yield json.loads(result)
        args_callback.on_args_end()

    @staticmethod
    async def _to_result(
        arg: AsyncIterator[str],
        max_model_completion_tokens: int,
        callback: ResultCallback,
    ) -> str:
        result = ""
        try:
            for _ in range(max_model_completion_tokens):
                token = await anext(arg)
                callback.on_result(token)
                result += token
            logger.warn(
                f"Max token count of {max_model_completion_tokens} exceeded in the reply"
            )
        except StopAsyncIteration:
            pass
        return result

    @staticmethod
    async def _execute_command(
        name: str,
        command: Command,
        args: AsyncIterator[JsonNode],
        chain_callback: ChainCallback,
    ) -> CommandResult:
        try:
            with chain_callback.command_callback() as command_callback:
                command_callback.on_command(name)
                args_list = [
                    arg
                    async for arg in CommandChain._to_args(
                        args, command_callback
                    )
                ]
                response = await command.execute(
                    args_list, command_callback.execution_callback()
                )
                command_callback.on_result(response)

                return {"status": Status.SUCCESS, "response": response.text}
        except Exception as e:
            logger.exception(f"Failed to execute command {name}")
            return {"status": Status.ERROR, "response": str(e)}

    @staticmethod
    async def _skip_text(stream: AsyncPeekable[str]):
        try:
            while True:
                char = await stream.apeek()
                if char == JsonObject.token():
                    break

                await anext(stream)
        except StopAsyncIteration:
            pass
