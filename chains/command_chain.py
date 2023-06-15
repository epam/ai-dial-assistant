import json
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import BaseMessage

from chains.base_chain import BaseChain
from protocol.command_result import CommandResultDict, Status, execute_command
from protocol.commands.end_dialog import EndDialog
from protocol.execution_context import ExecutionContext


class InvocationResult:
    def __init__(self, content: str, final_response: str | None):
        self.content = content
        self.final_response = final_response


class CommandChain(BaseChain):
    response_id: int

    def __init__(
        self,
        model: ChatOpenAI,
        name: str,
        init_messages: List[BaseMessage],
        resp_prompt: HumanMessagePromptTemplate,
        ctx: ExecutionContext,
        stop: List[str] | None = None,
    ):
        super().__init__(model, name, stop)
        self.init_messages = init_messages
        self.ctx = ctx
        self.resp_prompt = resp_prompt
        self.response_id = 1

    def run_chat(self) -> str:
        for message in self.init_messages:
            self.add_message(message)

        while True:
            invocation = self.run()
            self.add_message(invocation)

            result = self._execute_commands(invocation.content)
            message = self.resp_prompt.format(responses=result.content)
            self.add_message(message)

            if result.final_response is not None:
                return result.final_response

    def _execute_commands(self, invocation: str) -> InvocationResult:
        responses: List[CommandResultDict] = []
        final_response: str | None = None

        try:
            commands = self.ctx.parse_commands(invocation)
            for command in commands:
                responses.append(execute_command(command, self.response_id).to_dict())
                self.response_id += 1

                if isinstance(command, EndDialog):
                    final_response = command.response
                    break

        except Exception as e:
            final_response = None
            responses = [
                CommandResultDict(
                    id=self.response_id, status=Status.ERROR, response=str(e)
                )
            ]
            self.response_id += 1

        return InvocationResult(json.dumps({"responses": responses}), final_response)
