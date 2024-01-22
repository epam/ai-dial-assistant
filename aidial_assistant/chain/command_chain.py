import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Tuple, cast

from openai import BadRequestError

from aidial_assistant.application.prompts import ENFORCE_JSON_FORMAT_TEMPLATE
from aidial_assistant.chain.callbacks.args_callback import ArgsCallback
from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.chain.command_result import (
    CommandInvocation,
    CommandResult,
    Status,
    commands_to_text,
    responses_to_text,
)
from aidial_assistant.chain.dialogue import Dialogue, DialogueTurn
from aidial_assistant.chain.history import History
from aidial_assistant.chain.model_response_reader import (
    AssistantProtocolException,
    CommandsReader,
    skip_to_json_start,
)
from aidial_assistant.commands.base import (
    Command,
    CommandConstructor,
    FinalCommand,
)
from aidial_assistant.json_stream.chunked_char_stream import ChunkedCharStream
from aidial_assistant.json_stream.exceptions import JsonParsingException
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_parser import JsonParser, string_node
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.model.model_client import (
    ChatCompletionMessageParam,
    ModelClient,
    ModelClientRequest,
)
from aidial_assistant.utils.stream import CumulativeStream

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRY_COUNT = 3

# Some relatively large number to avoid CxSAST warning about potential DoS attack.
# Later, the upper limit will be provided by the DIAL Core (proxy).
MAX_MODEL_COMPLETION_CHUNKS = 32000

CommandDict = dict[str, CommandConstructor]


class LimitExceededException(Exception):
    pass


class ModelRequestLimiter(ABC):
    @abstractmethod
    async def verify_limit(self, messages: list[ChatCompletionMessageParam]):
        pass


class CommandChain:
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        command_dict: CommandDict,
        max_completion_tokens: int | None = None,
        max_retry_count: int = DEFAULT_MAX_RETRY_COUNT,
    ):
        self.name = name
        self.model_client = model_client
        self.command_dict = command_dict
        self.max_completion_tokens = max_completion_tokens
        self.max_retry_count = max_retry_count

    def _log_message(self, role: str, content: str | None):
        logger.debug(f"[{self.name}] {role}: {content or ''}")

    def _log_messages(self, messages: list[ChatCompletionMessageParam]):
        if logger.isEnabledFor(logging.DEBUG):
            for message in messages:
                self._log_message(message["role"], message.get("content"))

    async def run_chat(
        self,
        history: History,
        callback: ChainCallback,
        model_request_limiter: ModelRequestLimiter | None = None,
    ):
        dialogue = Dialogue()
        try:
            messages = history.to_protocol_messages()
            while True:
                dialogue_turn = await self._run_with_protocol_failure_retries(
                    callback,
                    messages + dialogue.messages,
                    model_request_limiter,
                )

                if dialogue_turn is None:
                    break

                dialogue.append(dialogue_turn)
        except (JsonParsingException, AssistantProtocolException):
            messages = (
                history.to_best_effort_messages(
                    "The next constructed API request is incorrect.",
                    dialogue,
                )
                if not dialogue.is_empty()
                else history.to_user_messages()
            )
            await self._generate_result(messages, callback)
        except (BadRequestError, LimitExceededException) as e:
            if dialogue.is_empty() or (
                isinstance(e, BadRequestError) and e.code == "429"
            ):
                raise

            # Assuming the context length is exceeded
            dialogue.pop()
            # TODO: Limit the error message size. The error message should not exceed reserved assistant overheads.
            await self._generate_result(
                history.to_best_effort_messages(str(e), dialogue), callback
            )

    async def _run_with_protocol_failure_retries(
        self,
        callback: ChainCallback,
        messages: list[ChatCompletionMessageParam],
        model_request_limiter: ModelRequestLimiter | None = None,
    ) -> DialogueTurn | None:
        last_error: Exception | None = None
        try:
            self._log_messages(messages)
            retries = Dialogue()
            while True:
                all_messages = self._reinforce_json_format(
                    messages + retries.messages
                )
                if model_request_limiter:
                    await model_request_limiter.verify_limit(all_messages)

                chunk_stream = CumulativeStream(
                    self.model_client.agenerate(
                        ModelClientRequest(
                            messages=all_messages,
                            max_tokens=self.max_completion_tokens,
                        )
                    )
                )
                try:
                    commands, responses = await self._run_commands(
                        chunk_stream, callback
                    )

                    if responses:
                        request_text = commands_to_text(commands)
                        response_text = responses_to_text(responses)

                        callback.on_state(request_text, response_text)
                        return DialogueTurn(
                            assistant_message=request_text,
                            user_message=response_text,
                        )

                    break
                except (JsonParsingException, AssistantProtocolException) as e:
                    logger.exception("Failed to process model response")

                    retry_count = retries.dialogue_turn_count()
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
                        DialogueTurn(
                            assistant_message=chunk_stream.buffer,
                            user_message="Failed to parse JSON commands: "
                            + str(e),
                        )
                    )
                finally:
                    self._log_message("assistant", chunk_stream.buffer)
        except (BadRequestError, LimitExceededException) as e:
            if last_error:
                # Retries can increase the prompt size, which may lead to token overflow.
                # Thus, if the original error was a protocol error, it should be thrown instead.
                raise last_error

            callback.on_error("Error", str(e))

            raise

    async def _run_commands(
        self, chunk_stream: AsyncIterator[str], callback: ChainCallback
    ) -> Tuple[list[CommandInvocation], list[CommandResult]]:
        char_stream = ChunkedCharStream(chunk_stream)
        await skip_to_json_start(char_stream)

        root_node = await JsonParser().parse(char_stream)
        commands: list[CommandInvocation] = []
        responses: list[CommandResult] = []
        request_reader = CommandsReader(root_node)
        async for invocation in request_reader.parse_invocations():
            command_name = await invocation.parse_name()
            command = self._create_command(command_name)
            args = await invocation.parse_args()
            if isinstance(command, FinalCommand):
                if len(responses) > 0:
                    continue
                message = string_node(await args.get("message"))
                await CommandChain._to_result(
                    message
                    if isinstance(message, JsonString)
                    else message.to_chunks(),
                    callback.result_callback(),
                )
                break
            else:
                response = await CommandChain._execute_command(
                    command_name, command, args, callback
                )

                commands.append(
                    cast(CommandInvocation, invocation.node.value())
                )
                responses.append(response)

        return commands, responses

    def _create_command(self, name: str) -> Command:
        if name not in self.command_dict:
            raise AssistantProtocolException(
                f"The command '{name}' is expected to be one of {list(self.command_dict.keys())}"
            )

        return self.command_dict[name]()

    async def _generate_result(
        self,
        messages: list[ChatCompletionMessageParam],
        callback: ChainCallback,
    ):
        stream = self.model_client.agenerate(
            ModelClientRequest(messages=messages)
        )

        await CommandChain._to_result(stream, callback.result_callback())

    @staticmethod
    def _reinforce_json_format(
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        last_message = messages[-1].copy()
        last_message["content"] = ENFORCE_JSON_FORMAT_TEMPLATE.render(
            response=last_message.get("content", "")
        )
        return messages[:-1] + [last_message]

    @staticmethod
    async def _to_args(
        args: JsonObject, args_callback: ArgsCallback
    ) -> dict[str, Any]:
        args_callback.on_args_start()
        result = ""
        async for chunk in args.to_chunks():
            args_callback.on_args_chunk(chunk)
            result += chunk
        parsed_args = json.loads(result)
        args_callback.on_args_end()

        return parsed_args

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
        args: JsonObject,
        chain_callback: ChainCallback,
    ) -> CommandResult:
        try:
            with chain_callback.command_callback() as command_callback:
                command_callback.on_command(name)
                response = await command.execute(
                    await CommandChain._to_args(
                        args, command_callback.args_callback()
                    ),
                    command_callback.execution_callback(),
                )
                command_callback.on_result(response)

                return {"status": Status.SUCCESS, "response": response.text}
        except Exception as e:
            logger.exception(f"Failed to execute command {name}")
            return {"status": Status.ERROR, "response": str(e)}
