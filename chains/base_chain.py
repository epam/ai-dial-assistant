from abc import ABC
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import AIMessage, BaseMessage

from utils.printing import print_base_message
from utils.token_counter import TokenCounter


class BaseChain(ABC):
    history: List[BaseMessage]

    def __init__(
        self,
        model: ChatOpenAI,
        name: str,
        stop: List[str] | None = None,
    ):
        self.model = model
        self.name = name
        self.stop = stop
        self.token_counter = TokenCounter()
        self.history = []

    def _session_prefix(self) -> str:
        return f"[{self.name}] "

    def _print_session_prefix(self, message: BaseMessage):
        prefix_message = message.copy()
        prefix_message.content = self._session_prefix()
        print_base_message(prefix_message, end="")

    def add_message(self, message: BaseMessage):
        self.history.append(message)

        if isinstance(message, AIMessage) and self.model.streaming:
            return

        self._print_session_prefix(message)
        print_base_message(message)

    def run(self) -> AIMessage:
        return self.__call__(self.history)

    def __call__(self, prompt: List[BaseMessage]) -> AIMessage:
        if self.model.streaming:
            self._print_session_prefix(AIMessage(content=""))

        llm_result = self.model.generate_prompt(
            [ChatPromptValue(messages=prompt)], self.stop
        )
        content = llm_result.generations[0][-1].text
        response = AIMessage(content=content)
        self.token_counter.update(
            self.model.model_name,
            self.model.get_num_tokens_from_messages(prompt),
            self.model.get_num_tokens_from_messages([response]),
        )
        self.token_counter.print(prefix=self._session_prefix())
        return response
