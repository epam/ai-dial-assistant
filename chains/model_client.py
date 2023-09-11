from abc import ABC
from asyncio import Queue, create_task
from typing import Any, List, Optional, Union, AsyncIterator
from uuid import UUID

import openai
from aiohttp import ClientSession
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, LLMResult
from typing_extensions import override


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
        buffer_size: int,
        stop: List[str] | None = None,
    ):
        self.model = model
        self.stop = stop
        self.buffer_size = buffer_size

    def generate(self, messages: List[BaseMessage]) -> AIMessage:
        llm_result = self.model.generate([messages], self.stop)
        content = llm_result.generations[0][-1].text
        response = AIMessage(content=content)
        return response

    async def agenerate(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        async with ClientSession(read_bufsize=self.buffer_size) as session:
            openai.aiosession.set(session)
            queue = Queue[str | None]()
            callback = AsyncChunksCallbackHandler(queue)
            producer = create_task(
                self.model.agenerate([messages], self.stop, callbacks=[callback])
            )
            while True:
                token = await callback.queue.get()
                if token is None:
                    break
                if isinstance(token, BaseException):
                    raise token

                yield token

            await producer
