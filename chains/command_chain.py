import json
import re
from abc import ABC, abstractmethod
from typing import List

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage

from chains.model_client import ModelClient
from protocol.command_result import (
    CommandResultDict,
    Status,
    execute_command,
    responses_to_text,
)
from protocol.commands.end_dialog import EndDialog
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import ExecutionContext
from utils.printing import print_base_message, print_exception


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

        if (
            self.say_or_ask
            or f'"{SayOrAsk.token()}"' in self.invocation[self.position :]
        ):
            if not self.say_or_ask:
                self.say_or_ask = True
            if not self.started:
                match = re.search(
                    r'"args"\s*:\s*\[\s*"(?P<message>(?:[^"]|\\.)*)$',
                    self.invocation[self.position :],
                )
                if match:
                    self.callback.on_message(text=match.group("message"))
                    self.started = True
            else:
                match = re.search(r'^(?P<message>([^"]|\\.)*)', token)
                if match:
                    message = match.group("message")
                    self.callback.on_message(text=message)
                    if len(message) < len(token):
                        self.started = False
                        self.position = len(self.invocation) - len(token) + len(message)
        else:
            if not self.another_command:
                match = re.search(
                    r'"command"\s*:\s*"(?P<command>[^"]+)"',
                    self.invocation[self.position :],
                )
                if match:
                    self.callback.on_message(
                        text=">>>COMMAND:" + match.group("command") + "("
                    )
                    self.another_command = True

            if not self.started:
                match = re.search(
                    r'"args"\s*:\s*\[\s*(?P<message>[^]]*)$',
                    self.invocation[self.position :],
                )
                if match:
                    self.callback.on_message(text=match.group("message"))
                    self.started = True
            else:
                match = re.search(r"^(?P<message>[^]]*)", token)
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
        self.name = name

    def _session_prefix(self) -> str:
        return f"[{self.name}] "

    def _print_session_prefix(self, message: BaseMessage):
        prefix_message = message.copy()
        prefix_message.content = self._session_prefix()
        print_base_message(prefix_message, end="")

    def run_chat(
        self, history: List[BaseMessage], callback: ChainCallback | None = None
    ) -> str:
        for message in history:
            self._print_session_prefix(message)
            print_base_message(message)

        callback.on_message(role="assistant") if callback else None
        while True:
            stream = self.model_client.stream(history)
            publisher = MessagePublisher(callback)
            self._print_session_prefix(AIMessage(content=""))
            for token in stream:
                publisher.stream_message_if_need(token.content)

            invocation = AIMessage(content=publisher.invocation)
            history.append(invocation)

            result = self._execute_commands(invocation.content, callback)
            message = self.resp_prompt.format(responses=result.content)

            self._print_session_prefix(message)
            print_base_message(message)
            history.append(message)

            if result.final_response is not None:
                return result.final_response

    def _execute_commands(
        self, invocation: str, callback: ChainCallback | None
    ) -> InvocationResult:
        responses: List[CommandResultDict] = []
        final_response: str | None = None

        try:
            commands = self.ctx.parse_commands(invocation)
            for command in commands:
                self.response_id += 1

                if isinstance(command, EndDialog):
                    final_response = command.response
                    response = CommandResultDict(
                        id=self.response_id,
                        status=Status.SUCCESS,
                        response=final_response,
                    )
                    responses.append(response)

                elif isinstance(command, SayOrAsk):
                    # It's the last message in the main session
                    final_response = "[DONE]"
                    response = CommandResultDict(
                        id=self.response_id,
                        status=Status.SUCCESS,
                        response=final_response,
                    )
                    responses.append(response)
                    callback.on_end() if callback else None

                else:
                    response = execute_command(command, self.response_id).to_dict()
                    responses.append(response)

                    text = ">>>RESPONSE:" + json.dumps(response["response"]) + "<<<\n"
                    callback.on_message(text=text) if callback else None

        except Exception as e:
            print_exception()
            final_response = None
            text = ">>>ERROR:" + str(e) + "<<<\n"
            callback.on_message(text=text) if callback else None
            responses = [
                CommandResultDict(
                    id=self.response_id, status=Status.ERROR, response=str(e)
                )
            ]
            self.response_id += 1

        return InvocationResult(responses_to_text(responses), final_response)
