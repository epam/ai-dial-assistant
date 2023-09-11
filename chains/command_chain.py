import json
from typing import List, AsyncIterator, Any

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from chains import logger
from chains.callbacks.chain_callback import ChainCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback
from chains.json_stream.json_node import JsonNode
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_parser import JsonParser
from chains.json_stream.json_string import JsonString
from chains.json_stream.tokenator import Tokenator, AsyncPeekable
from chains.model_client import ModelClient
from chains.request_reader import RequestReader
from protocol.command_result import responses_to_text, CommandResult, Status
from protocol.commands.base import FinalCommand, Command
from protocol.execution_context import ExecutionContext


class TextCollector(AsyncIterator[str]):
    def __init__(self, stream: AsyncIterator[str]):
        self.stream = stream
        self.buffer = ""

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        chunk = await anext(self.stream)
        self.buffer += chunk
        return chunk


MAX_MESSAGE_COUNT = 20
MAX_RETRY_COUNT = 2


class CommandChain:
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        resp_prompt: HumanMessagePromptTemplate,
        ctx: ExecutionContext,
    ):
        self.model_client = model_client
        self.ctx = ctx
        self.resp_prompt = resp_prompt
        self.name = name

    def _print(self, message: BaseMessage) -> BaseMessage:
        logger.debug(f"[{self.name}] {message}")
        return message

    async def run_chat(
        self, history: List[BaseMessage], callback: ChainCallback
    ) -> str:
        for message in history:
            self._print(message)

        await callback.on_start()
        message_count = 0
        retry_count = 0
        while True:
            if message_count >= MAX_MESSAGE_COUNT:
                raise Exception(f"Max message count of {MAX_MESSAGE_COUNT} exceeded")
            message_count += 1

            token_stream = TextCollector(
                self.model_client.agenerate(self._reinforce_last_message(history))
            )
            tokenator = Tokenator(token_stream)
            await CommandChain._skip_text(tokenator)
            parsing_content = await JsonParser.parse(tokenator)

            try:
                responses: List[CommandResult] = []
                root_node = await parsing_content.root.node()
                request_reader = RequestReader(root_node)
                final_command_name = None
                async for invocation in request_reader.parse_invocations():
                    command_name = await invocation.parse_name()
                    command = self.ctx.create_command(command_name)
                    args = invocation.parse_args()
                    if isinstance(command, FinalCommand):
                        final_command_name = command_name
                        if len(responses) > 0:
                            continue
                        arg = await anext(args)
                        result = await CommandChain._to_result(
                            arg if isinstance(arg, JsonString) else arg.to_string_tokens(),  # type: ignore
                            # Some relatively large number to avoid CxSAST warning about potential DoS attack.
                            # Later, the upper limit will be provided by the DIAL Core (proxy).
                            32000,
                            callback.result_callback(),
                        )
                        await callback.on_end()
                        self._print(AIMessage(content=json.dumps(root_node.value())))
                        return result
                    else:
                        response = await CommandChain._execute_command(
                            command_name, command, args, callback.command_callback()
                        )

                        responses.append(response)

                if len(responses) == 0:
                    # Assume the model has nothing to say
                    await callback.on_end()
                    self._print(AIMessage(content=json.dumps(root_node.value())))
                    return ""

                fixed_model_response = json.dumps(
                    {
                        # Remove final command when it's generated before the result is known to the model
                        "commands": [
                            c
                            for c in root_node.value()["commands"]
                            if c["command"] != final_command_name
                        ]
                    }
                )

                history.append(self._print(AIMessage(content=fixed_model_response)))

                response_text = responses_to_text(responses)
                history.append(self._print(HumanMessage(content=response_text)))
                await callback.on_state(fixed_model_response, response_text)
                retry_count = 0
            except Exception as e:
                logger.exception("Failed to process model response")
                await callback.on_error(
                    "Error" if retry_count == 0 else f"Error (retry {retry_count})", e
                )

                retry_count += 1
                if retry_count > MAX_RETRY_COUNT:
                    raise e

                await parsing_content.finish_parsing()

                if token_stream.buffer:
                    history.append(self._print(AIMessage(content=token_stream.buffer)))
                    history.append(
                        self._print(HumanMessage(content=json.dumps({"error": str(e)})))
                    )

    def _reinforce_last_message(self, history: list[BaseMessage]) -> list[BaseMessage]:
        return history[:-1] + [self.resp_prompt.format(response=history[-1].content)]

    @staticmethod
    async def _to_args(
        args: AsyncIterator[JsonNode], callback: CommandCallback
    ) -> AsyncIterator[Any]:
        args_callback = callback.args_callback()
        await args_callback.on_args_start()
        async for arg in args:
            arg_callback = args_callback.arg_callback()
            await arg_callback.on_arg_start()
            result = ""
            async for token in arg.to_string_tokens():  # type: ignore
                await arg_callback.on_arg(token)
                result += token
            await arg_callback.on_arg_end()
            yield json.loads(result)
        await args_callback.on_args_end()

    @staticmethod
    async def _to_result(
        arg: AsyncIterator[str],
        max_model_completion_tokens: int,
        callback: ResultCallback,
    ) -> str:
        result = ""
        token_count = 0
        await callback.on_start()
        async for token in arg:
            token_count += 1
            if token_count > max_model_completion_tokens:
                raise Exception(
                    f"Max token count of {max_model_completion_tokens} exceeded in the reply"
                )
            await callback.on_result(token)
            result += token
        await callback.on_end()
        return result

    @staticmethod
    async def _execute_command(
        name: str,
        command: Command,
        args: AsyncIterator[JsonNode],
        callback: CommandCallback,
    ) -> CommandResult:
        try:
            await callback.on_command(name)
            args_list = [arg async for arg in CommandChain._to_args(args, callback)]
            response = await command.execute(args_list, callback.execution_callback())
            await callback.on_result(response)

            return {"status": Status.SUCCESS, "response": response.text}
        except Exception as e:
            logger.exception(f"Failed to execute command {name}")
            await callback.on_error(e)
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
