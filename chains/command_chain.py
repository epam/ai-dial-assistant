import threading
from abc import ABC, abstractmethod
from typing import List, Iterable, Iterator

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage

from chains.json_object import JsonObject, Tokenator
from chains.model_client import ModelClient
from protocol.command_result import responses_to_text, execute_command, Status, CommandResult
from protocol.commands.base import FinalCommand, ExecutionCallback
from protocol.execution_context import ExecutionContext
from utils.printing import print_base_message


class ArgCallback(ABC):
    @abstractmethod
    def on_arg_start(self):
        pass

    @abstractmethod
    def on_arg(self, token: str):
        pass

    @abstractmethod
    def on_arg_end(self):
        pass


class ArgsCallback(ABC):
    @abstractmethod
    def on_args_start(self):
        pass

    @abstractmethod
    def arg_callback(self) -> ArgCallback:
        pass

    @abstractmethod
    def on_args_end(self):
        pass


class CommandCallback(ABC):
    @abstractmethod
    def on_command(self, command: str):
        pass

    @abstractmethod
    def args_callback(self) -> ArgsCallback:
        pass

    @abstractmethod
    def execution_callback(self) -> ExecutionCallback:
        pass

    @abstractmethod
    def on_result(self, response):
        pass


class ResultCallback(ABC):
    @abstractmethod
    def on_start(self):
        pass

    @abstractmethod
    def on_result(self, token):
        pass

    @abstractmethod
    def on_end(self):
        pass


class ChainCallback(ABC):
    @abstractmethod
    def on_start(self):
        pass

    @abstractmethod
    def command_callback(self) -> CommandCallback:
        pass

    @abstractmethod
    def on_end(self, error: str | None = None):
        pass

    @abstractmethod
    def on_ai_message(self, message: str):
        pass

    @abstractmethod
    def on_human_message(self, message: str):
        pass

    @abstractmethod
    def result_callback(self) -> ResultCallback:
        pass


class BufferedIterator(Iterator[str]):
    def __init__(self, stream: Iterator[str]):
        self.stream = stream
        self.buffer = ""

    def __next__(self) -> str:
        chunk = next(self.stream)
        self.buffer += chunk
        return chunk


class InvocationResult:
    def __init__(self, content: str, final_response: str | None):
        self.content = content
        self.final_response = final_response


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

    def run_chat(
        self, history: List[BaseMessage], callback: ChainCallback
    ) -> str:
        for message in history:
            self._print_session_prefix(message)
            print_base_message(message)

        callback.on_start()
        while True:
            token_stream = BufferedIterator(map(lambda m: m.content, self.model_client.stream(history)))
            json_object = JsonObject()
            thread = threading.Thread(target=json_object.parse, args=[Tokenator(token_stream)])
            thread.start()
            self._print_session_prefix(AIMessage(content=""))

            responses: List[CommandResult] = []
            invocations = json_object["commands"]
            for invocation in invocations:
                command_name = ''.join(invocation["command"])
                try:
                    command = self.ctx.create_command(command_name)
                    args = invocation["args"]
                    if isinstance(command, FinalCommand):
                        result = CommandChain._to_result(next(args), callback.result_callback())
                        callback.on_end()
                        return result
                    else:
                        command_callback = callback.command_callback()
                        command_callback.on_command(command_name)
                        response = execute_command(
                            command,
                            list(CommandChain._to_args(args, command_callback)),
                            command_callback.execution_callback())
                        command_callback.on_result(response["status"] + ": " + response["response"])
                        responses.append(response)
                except Exception as e:
                    responses.append({"status": Status.ERROR, "response": str(e)})

            thread.join()

            callback.on_ai_message(token_stream.buffer)
            history.append(AIMessage(content=token_stream.buffer))

            response_text = self.resp_prompt.format(responses=responses_to_text(responses))
            self._print_session_prefix(response_text)
            print_base_message(response_text)
            callback.on_human_message(response_text.content)
            history.append(response_text)

    @staticmethod
    def _to_args(args: Iterable[Iterable[str]], callback: CommandCallback) -> Iterable[str]:
        args_callback = callback.args_callback()
        args_callback.on_args_start()
        for arg in args:
            arg_callback = args_callback.arg_callback()
            arg_callback.on_arg_start()
            result = ""
            for token in arg:
                arg_callback.on_arg(token)
                result += token
            arg_callback.on_arg_end()
            yield result
        args_callback.on_args_end()

    @staticmethod
    def _to_result(arg: Iterable[str], callback: ResultCallback) -> str:
        result = ""
        callback.on_start()
        for token in arg:
            callback.on_result(token)
            result += token
        callback.on_end()
        return result

