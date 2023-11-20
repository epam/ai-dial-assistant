import json
import logging
from typing import Any, AsyncIterator, Callable, Tuple

from aidial_sdk.chat_completion.request import Role
from openai import InvalidRequestError
from typing_extensions import override

from aidial_assistant.application.prompts import ENFORCE_JSON_FORMAT_TEMPLATE
from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.chain.command_result import (
    CommandResult,
    Status,
    responses_to_text,
)
from aidial_assistant.chain.dialogue import Dialogue
from aidial_assistant.chain.history import History
from aidial_assistant.chain.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.chain.model_response_reader import (
    AssistantProtocolException,
    CommandsReader,
    skip_to_json_start,
)
from aidial_assistant.commands.base import Command, FinalCommand
from aidial_assistant.json_stream.characterstream import CharacterStream
from aidial_assistant.json_stream.exceptions import JsonParsingException
from aidial_assistant.json_stream.json_node import JsonNode
from aidial_assistant.json_stream.json_parser import JsonParser
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.utils.stream import CumulativeStream

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRY_COUNT = 3

# Some relatively large number to avoid CxSAST warning about potential DoS attack.
# Later, the upper limit will be provided by the DIAL Core (proxy).
MAX_MODEL_COMPLETION_CHUNKS = 32000

CommandConstructor = Callable[[], Command]
CommandDict = dict[str, CommandConstructor]


class ModelExtraResultsCallback(ExtraResultsCallback):
    def __init__(self):
        self._discarded_messages: int | None = None

    @override
    def on_discarded_messages(self, discarded_messages: int):
        self._discarded_messages = discarded_messages

    @property
    def discarded_messages(self) -> int | None:
        return self._discarded_messages


class CommandChain:
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        command_dict: CommandDict,
        max_prompt_tokens: int | None = None,
        max_retry_count: int = DEFAULT_MAX_RETRY_COUNT,
    ):
        self.name = name
        self.model_client = model_client
        self.command_dict = command_dict
        self.max_prompt_tokens = max_prompt_tokens
        self.max_retry_count = max_retry_count

    def _log_message(self, role: Role, content: str):
        logger.debug(f"[{self.name}] {role.value}: {content}")

    def _log_messages(self, messages: list[Message]):
        if logger.isEnabledFor(logging.DEBUG):
            for message in messages:
                self._log_message(message.role, message.content)

    async def run_chat(self, history: History, callback: ChainCallback):
        dialogue = Dialogue()
        try:
            history = await self._trim_history(history, callback)
            messages = history.to_protocol_messages_with_system_message()
            while True:
                pair = await self._run_with_protocol_failure_retries(
                    callback, messages + dialogue.messages
                )

                if pair is None:
                    break

                dialogue.append(pair[0], pair[1])
        except (JsonParsingException, AssistantProtocolException):
            messages = (
                history.to_best_effort_messages(
                    "The next constructed API request is incorrect.",
                    dialogue,
                )
                if not dialogue.is_empty()
                else history.to_client_messages()
            )
            await self._generate_result(messages, callback)
        except InvalidRequestError as e:
            if dialogue.is_empty() or e.code == "429":
                raise

            # Assuming the context length is exceeded
            dialogue.pop()
            await self._generate_result(
                history.to_best_effort_messages(str(e), dialogue), callback
            )

    async def _run_with_protocol_failure_retries(
        self, callback: ChainCallback, messages: list[Message]
    ) -> Tuple[str, str] | None:
        last_error: Exception | None = None
        try:
            self._log_messages(messages)
            retries = Dialogue()
            while True:
                chunk_stream = CumulativeStream(
                    self.model_client.agenerate(
                        self._reinforce_json_format(messages + retries.messages)
                    )
                )
                try:
                    commands, responses = await self._run_commands(
                        chunk_stream, callback
                    )

                    if responses:
                        request_text = json.dumps({"commands": commands})
                        response_text = responses_to_text(responses)

                        callback.on_state(request_text, response_text)
                        return request_text, response_text

                    return None
                except (JsonParsingException, AssistantProtocolException) as e:
                    logger.exception("Failed to process model response")

                    retry_count = len(retries.messages) // 2
                    callback.on_error(
                        "Error"
                        if retry_count == 0
                        else f"Error (retry {retry_count})",
                        "The model failed to construct addon request.",
                    )

                    if retry_count >= self.max_retry_count:
                        raise

                    last_error = e
                    retries.append(
                        chunk_stream.buffer, json.dumps({"error": str(e)})
                    )
                finally:
                    self._log_message(Role.ASSISTANT, chunk_stream.buffer)
        except InvalidRequestError as e:
            if last_error:
                # Retries can increase the prompt size, which may lead to token overflow.
                # Thus, if the original error was a protocol error, it should be thrown instead.
                raise last_error

            callback.on_error("Error", str(e))

            raise

    async def _run_commands(
        self, chunk_stream: AsyncIterator[str], callback: ChainCallback
    ) -> Tuple[list[dict[str, Any]], list[CommandResult]]:
        char_stream = CharacterStream(chunk_stream)
        await skip_to_json_start(char_stream)

        async with JsonParser.parse(char_stream) as root_node:
            commands: list[dict[str, Any]] = []
            responses: list[CommandResult] = []
            request_reader = CommandsReader(root_node)
            async for invocation in request_reader.parse_invocations():
                command_name = await invocation.parse_name()
                command = self._create_command(command_name)
                args = invocation.parse_args()
                if isinstance(command, FinalCommand):
                    if len(responses) > 0:
                        continue
                    message = await anext(args)
                    await CommandChain._to_result(
                        message
                        if isinstance(message, JsonString)
                        else message.to_string_chunks(),
                        callback.result_callback(),
                    )
                    break
                else:
                    response = await CommandChain._execute_command(
                        command_name, command, args, callback
                    )

                    commands.append(invocation.node.value())
                    responses.append(response)

            return commands, responses

    def _create_command(self, name: str) -> Command:
        if name not in self.command_dict:
            raise Exception(
                f"The command '{name}' is expected to be one of {[*self.command_dict.keys()]}"
            )

        return self.command_dict[name]()

    async def _generate_result(
        self, messages: list[Message], callback: ChainCallback
    ):
        stream = self.model_client.agenerate(messages)

        await CommandChain._to_result(stream, callback.result_callback())

    async def _trim_history(
        self, history: History, callback: ChainCallback
    ) -> History:
        if self.max_prompt_tokens is None:
            return history

        extra_results_callback = ModelExtraResultsCallback()
        stream = self.model_client.agenerate(
            history.to_protocol_messages(),
            extra_results_callback,
            max_prompt_tokens=self.max_prompt_tokens,
            max_tokens=1,
        )
        try:
            async for _ in stream:
                pass
        except ReasonLengthException:
            # Expected for max_tokens=1
            pass

        if extra_results_callback.discarded_messages:
            old_size = history.user_message_count()
            history = history.trim(extra_results_callback.discarded_messages)
            callback.on_discarded_messages(
                old_size - history.user_message_count()
            )
        else:
            callback.on_discarded_messages(0)

        return history

    @staticmethod
    def _reinforce_json_format(messages: list[Message]) -> list[Message]:
        last_message = messages[-1]
        return messages[:-1] + [
            Message(
                role=last_message.role,
                content=ENFORCE_JSON_FORMAT_TEMPLATE.render(
                    response=last_message.content
                ),
            ),
        ]

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
            async for chunk in arg.to_string_chunks():
                arg_callback.on_arg(chunk)
                result += chunk
            arg_callback.on_arg_end()
            yield json.loads(result)
        args_callback.on_args_end()

    @staticmethod
    async def _to_result(stream: AsyncIterator[str], callback: ResultCallback):
        try:
            for _ in range(MAX_MODEL_COMPLETION_CHUNKS):
                chunk = await anext(stream)
                callback.on_result(chunk)
            logger.warning(
                f"Max chunk count of {MAX_MODEL_COMPLETION_CHUNKS} exceeded in the reply"
            )
        except StopAsyncIteration:
            pass

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
