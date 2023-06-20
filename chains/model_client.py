import threading
from abc import ABC
from queue import Queue
from typing import Any, Generator, List, Optional, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import AIMessage, BaseMessage, LLMResult
from typing_extensions import override

from utils.token_counter import TokenCounter

ERROR_PREFIX = "error:"
DATA_PREFIX = "data:"
END_TOKEN = "[DONE]"


class ChunksCallback(BaseCallbackHandler):
    queue = Queue[str]()

    @override
    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.queue.put(DATA_PREFIX + token)

    @override
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.queue.put(text)

    @override
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.queue.put(ERROR_PREFIX + str(error))

    @override
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.queue.put(END_TOKEN)


class ModelClient(ABC):
    def __init__(
        self,
        model: ChatOpenAI,
        stop: List[str] | None = None,
    ):
        self.model = model
        self.stop = stop
        self.token_counter = TokenCounter()

    def generate(self, prompt: List[BaseMessage]) -> AIMessage:
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
        self.token_counter.print()
        return response

    def _generate(self, prompt: List[BaseMessage], callback: ChunksCallback):
        self.model.generate_prompt(
            [ChatPromptValue(messages=prompt)], self.stop, callbacks=[callback]
        )

    def stream(self, prompt: List[BaseMessage]) -> Generator[AIMessage, Any, None]:
        callback = ChunksCallback()
        thread = threading.Thread(target=self._generate, args=(prompt, callback))
        thread.start()
        content = ""
        while True:
            item = callback.queue.get()
            if item == END_TOKEN:
                break
            if item.startswith(ERROR_PREFIX):
                raise Exception(item[len(ERROR_PREFIX) :])

            token = item[len(DATA_PREFIX) :]
            content += token
            yield AIMessage(content=token)

        thread.join()
        self.token_counter.update(
            self.model.model_name,
            self.model.get_num_tokens_from_messages(prompt),
            self.model.get_num_tokens_from_messages([AIMessage(content=content)]),
        )
        self.token_counter.print()
