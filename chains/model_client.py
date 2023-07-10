from abc import ABC
from asyncio import Queue, create_task
from typing import Any, List, Optional, Union, AsyncIterator
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, LLMResult
from typing_extensions import override

from utils.token_counter import TokenCounter


class AsyncChunksCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: Queue[str | BaseException | None]):
        self.queue = queue

    @override
    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        await self.queue.put(token)

    @override
    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        await self.queue.put(error)

    @override
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        await self.queue.put(None)


class ModelClient(ABC):
    def __init__(
        self,
        model: ChatOpenAI,
        stop: List[str] | None = None,
    ):
        self.model = model
        self.stop = stop
        self.token_counter = TokenCounter()

    def generate(self, messages: List[BaseMessage]) -> AIMessage:
        llm_result = self.model.generate([messages], self.stop)
        content = llm_result.generations[0][-1].text
        response = AIMessage(content=content)
        self.token_counter.update(
            self.model.model_name,
            self.model.get_num_tokens_from_messages(messages),
            self.model.get_num_tokens_from_messages([response]),
        )
        self.token_counter.print()
        return response

    async def agenerate(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        queue = Queue[str | None]()
        callback = AsyncChunksCallbackHandler(queue)
        producer = create_task(self.model.agenerate([messages], self.stop, callbacks=[callback]))
        content = ""
        while True:
            token = await callback.queue.get()
            if token is None:
                break
            if isinstance(token, BaseException):
                raise token

            content += token
            yield token

        await producer
        # self.token_counter.update(
        #     self.model.model_name,
        #     self.model.get_num_tokens_from_messages(messages),
        #     self.model.get_num_tokens_from_messages([AIMessage(content=content)]),
        # )
        # self.token_counter.print()
