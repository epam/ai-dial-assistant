import json
import traceback
from typing import List, Iterable, AsyncIterator, Any

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from chains.callbacks.chain_callback import ChainCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback
from chains.json_stream.json_node import JsonNode
from chains.json_stream.json_parser import JsonParser, object_node, array_node, string_node
from chains.json_stream.tokenator import Tokenator
from chains.model_client import ModelClient
from protocol.command_result import responses_to_text, execute_command, Status, CommandResult
from protocol.commands.base import FinalCommand
from protocol.execution_context import ExecutionContext
from utils.printing import print_base_message
from utils.text import join_string


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

    def _session_prefix(self) -> str:
        return f"[{self.name}] "

    def _print_session_prefix(self, message: BaseMessage):
        prefix_message = message.copy()
        prefix_message.content = self._session_prefix()
        print_base_message(prefix_message, end="")

    async def run_chat(self, history: List[BaseMessage], callback: ChainCallback) -> str:
        for message in history:
            self._print_session_prefix(message)
            print_base_message(message)

        await callback.on_start()
        while True:
            self._print_session_prefix(AIMessage(content=""))
            token_stream = TextCollector(self.model_client.agenerate(history))

            try:
                responses: List[CommandResult] = []
                parsing_content = await JsonParser.parse(Tokenator(token_stream))
                invocations = await object_node(parsing_content.node).get("commands")
                async for invocation in array_node(invocations):
                    invocation_object = object_node(invocation)
                    command_name = await join_string(string_node(await invocation_object.get("command")))
                    command = self.ctx.create_command(command_name)
                    args = array_node(await invocation_object.get("args"))
                    if isinstance(command, FinalCommand):
                        arg = await anext(args)
                        result = await CommandChain._to_result(string_node(arg), callback.result_callback())
                        await callback.on_end()
                        return result
                    else:
                        command_callback = callback.command_callback()
                        await command_callback.on_command(command_name)
                        response = await execute_command(
                            command,
                            [arg async for arg in CommandChain._to_args(args, command_callback)],
                            command_callback.execution_callback())
                        await command_callback.on_result(response["response"])
                        responses.append(response)

                await parsing_content.finish_parsing()

                await callback.on_ai_message(token_stream.buffer)
                history.append(AIMessage(content=token_stream.buffer))

                response_text = responses_to_text(responses)
                response = self.resp_prompt.format(responses=response_text)
                self._print_session_prefix(response)
                print_base_message(response)
                await callback.on_human_message(response_text)
                history.append(response)
            except Exception as e:
                traceback.print_exc()
                await callback.on_error(e)
                history.append(AIMessage(content=token_stream.buffer))

                response = self.resp_prompt.format(responses=json.dumps({"error": str(e)}))
                self._print_session_prefix(response)
                print_base_message(response)
                history.append(response)

    @staticmethod
    async def _to_args(args: AsyncIterator[JsonNode], callback: CommandCallback) -> AsyncIterator[str]:
        args_callback = callback.args_callback()
        await args_callback.on_args_start()
        async for arg in args:
            arg_callback = args_callback.arg_callback()
            await arg_callback.on_arg_start()
            result = ""
            async for token in string_node(arg):
                await arg_callback.on_arg(token)
                result += token
            await arg_callback.on_arg_end()
            yield result
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
