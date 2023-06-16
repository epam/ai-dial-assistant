import json
import re
from abc import ABC, abstractmethod
from typing import List

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import BaseMessage, AIMessage

from chains.model_client import ModelClient
from protocol.command_result import CommandResultDict, Status, execute_command, CommandResult
from protocol.commands.end_dialog import EndDialog
from protocol.execution_context import ExecutionContext
from protocol.commands.say_or_ask import SayOrAsk
from utils.printing import print_base_message


class ChainCallback(ABC):
    @abstractmethod
    def on_message(self, role: str | None = None, text: str | None = None):
        pass

    @abstractmethod
    def on_end(self, error: str | None = None):
        pass


class MessagePublisher:
    invocation: str = ""
    started = False
    say_or_ask = False
    another_command = False
    position = 0

    def __init__(self, callback: ChainCallback | None):
        self.callback = callback

    def stream_message_if_need(self, token: str):
        self.invocation += token

        if self.callback is None:
            return

        if self.say_or_ask or f'"{SayOrAsk.token()}"' in self.invocation[self.position:]:
            if not self.say_or_ask:
                self.say_or_ask = True
            if not self.started:
                match = re.search(r'"args"\s*:\s*\[\s*"(?P<message>(?:[^"\\]|\\.)*)$', self.invocation[self.position:])
                if match:
                    self.callback.on_message(text=match.group("message"))
                    self.started = True
            else:
                match = re.search(r'^(?P<message>([^"\\]|\\.)*)', token)
                if match:
                    message = match.group("message")
                    self.callback.on_message(text=message)
                    if len(message) < len(token):
                        self.started = False
                        self.position = len(self.invocation) - len(token) + len(message)
        else:
            if not self.another_command:
                match = re.search(r'"command"\s*:\s*"(?P<command>[^"]+)"', self.invocation[self.position:])
                if match:
                    self.callback.on_message(text=">>>Run command: " + match.group("command") + "(")
                    self.another_command = True

            if not self.started:
                match = re.search(r'"args"\s*:\s*\[\s*(?P<message>[^]]*)$', self.invocation[self.position:])
                if match:
                    self.callback.on_message(text=match.group("message"))
                    self.started = True
            else:
                match = re.search(r'^(?P<message>[^]]*)', token)
                if match:
                    message = match.group("message")
                    self.callback.on_message(text=message)
                    if len(message) < len(token):
                        self.started = False
                        self.callback.on_message(text=")<<<\n")
                        self.position = len(self.invocation) - len(token) + len(message)
                        self.another_command = False


class InvocationResult:
    def __init__(self, content: str, final_response: str | None):
        self.content = content
        self.final_response = final_response


class CommandChain:
    response_id: int

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
        self.response_id = 1

    def run_chat(self, history: List[BaseMessage], callback: ChainCallback | None = None) -> str:
        for message in history:
            print_base_message(message)

        callback.on_message(role="assistant") if callback else None
        while True:
            stream = self.model_client.stream(history)
            publisher = MessagePublisher(callback)
            for token in stream:
                publisher.stream_message_if_need(token.content)

            # if publisher.stopped:
            #     callback.on_end() if callback else None
            #     return ""

            invocation = AIMessage(content=publisher.invocation)
            history.append(invocation)

            result = self._execute_commands(invocation.content, callback)
            message = self.resp_prompt.format(responses=result.content)
            history.append(message)

            if result.final_response is not None:
                callback.on_end() if callback else None
                return result.final_response

    def _execute_commands(self, invocation: str, callback: ChainCallback) -> InvocationResult:
        responses: List[CommandResultDict] = []
        final_response: str | None = None

        try:
            commands = self.ctx.parse_commands(invocation)
            for command in commands:
                self.response_id += 1

                if isinstance(command, EndDialog):
                    final_response = command.response
                    result_stub = CommandResult(command, "[DONE]", self.response_id, Status.SUCCESS).to_dict()
                    responses.append(result_stub)
                    break

                if command.token() == "say-or-ask":
                    final_response = "[DONE]"
                    result_stub = CommandResult(command, "[DONE]", self.response_id, Status.SUCCESS).to_dict()
                    responses.append(result_stub)
                    break

                # text = ">>>Run command: " + command.token() + json.dumps(command.get_args()).replace('[', '(').replace(']', ')') + "<<<\n"
                # callback.on_message(text=text) if callback else None

                responses.append(execute_command(command, self.response_id).to_dict())

            if final_response is None:
                for response in responses:
                    text = ">>>" + response["response"] + "<<<\n"
                    callback.on_message(text=text) if callback else None

        except Exception as e:
            final_response = None
            callback.on_message(text=">>>" + str(e) + "<<<\n") if callback else None
            responses = [
                CommandResultDict(
                    id=self.response_id, status=Status.ERROR, response=str(e)
                )
            ]
            self.response_id += 1

        return InvocationResult(json.dumps({"responses": responses}), final_response)
