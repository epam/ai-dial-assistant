import json
from typing import List, AsyncIterator, Any

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage

from chains.callbacks.chain_callback import ChainCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback
from chains.json_stream.json_node import JsonNode
from chains.json_stream.json_parser import JsonParser, string_node, to_string
from chains.json_stream.tokenator import Tokenator
from chains.model_client import ModelClient
from chains.request_parser import RequestParser
from protocol.command_result import responses_to_text, CommandResult, Status
from protocol.commands.base import FinalCommand, Command
from protocol.execution_context import ExecutionContext
from utils.printing import print_base_message, print_exception


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
        print_base_message(f"[{self.name}] ", message)
        return message

    async def run_chat(self, history: List[BaseMessage], callback: ChainCallback) -> str:
        for message in history:
            self._print(message)

        await callback.on_start()
        message_count = 0
        while True:
            if message_count >= MAX_MESSAGE_COUNT:
                await callback.on_end()
                raise Exception(f"Max message count of {MAX_MESSAGE_COUNT} exceeded")
            message_count += 1

            token_stream = TextCollector(self.model_client.agenerate(history))
            parsing_content = await JsonParser.parse(Tokenator(token_stream))

            try:
                responses: List[CommandResult] = []
                request_parser = RequestParser(await parsing_content.root.node())
                async for invocation in request_parser.parse_invocations():
                    command_name = await invocation.parse_name()
                    command = self.ctx.create_command(command_name)
                    args = invocation.parse_args()
                    if isinstance(command, FinalCommand):
                        arg = await anext(args)
                        result = await CommandChain._to_result(string_node(arg), callback.result_callback())
                        await callback.on_end()
                        return result
                    else:
                        response = await CommandChain._execute_command(
                            command_name, command, args, callback.command_callback())

                        responses.append(response)

                await parsing_content.finish_parsing()

                history.append(self._print(AIMessage(content=token_stream.buffer)))

                response_text = responses_to_text(responses)
                history.append(self._print(self.resp_prompt.format(responses=response_text)))

                await callback.on_state(token_stream.buffer, response_text)
            except Exception as e:
                print_exception()
                await callback.on_error(e)

                await parsing_content.finish_parsing()
                history.append(self._print(AIMessage(content=token_stream.buffer)))
                history.append(self._print(self.resp_prompt.format(responses=json.dumps({"error": str(e)}))))

    @staticmethod
    async def _to_args(args: AsyncIterator[JsonNode], callback: CommandCallback) -> AsyncIterator[Any]:
        args_callback = callback.args_callback()
        await args_callback.on_args_start()
        async for arg in args:
            arg_callback = args_callback.arg_callback()
            await arg_callback.on_arg_start()
            result = ""
            async for token in to_string(arg):
                await arg_callback.on_arg(token)
                result += token
            await arg_callback.on_arg_end()
            yield json.loads(result)
        await args_callback.on_args_end()

    @staticmethod
    async def _to_result(arg: AsyncIterator[str], callback: ResultCallback) -> str:
        result = ""
        await callback.on_start()
        async for token in arg:
            await callback.on_result(token)
            result += token
        await callback.on_end()
        return result

    @staticmethod
    async def _execute_command(name: str, command: Command, args: AsyncIterator[JsonNode], callback: CommandCallback):
        try:
            await callback.on_command(name)
            args = [arg async for arg in CommandChain._to_args(args, callback)]
            response = await command.execute(args, callback.execution_callback())
            await callback.on_result(response)

            return {"status": Status.SUCCESS, "response": response}
        except Exception as e:
            print_exception()
            await callback.on_error(e)
            return {"status": Status.ERROR, "response": str(e)}
