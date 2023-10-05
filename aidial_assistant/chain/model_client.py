from abc import ABC
from typing import Any, AsyncIterator, List

import openai
from aidial_sdk.chat_completion.request import Message
from aiohttp import ClientSession


class ModelClient(ABC):
    def __init__(
        self,
        model_args: dict[str, Any],
        buffer_size: int,
    ):
        self.model_args = model_args
        self.buffer_size = buffer_size

    async def agenerate(self, messages: List[Message]) -> AsyncIterator[str]:
        async with ClientSession(read_bufsize=self.buffer_size) as session:
            openai.aiosession.set(session)

            model_result = await openai.ChatCompletion.acreate(
                **self.model_args,
                messages=[
                    {
                        "role": m.role.value,
                        "content": m.content,
                    }
                    for m in messages
                ]
            )

            async for chunk in model_result:  # type: ignore
                text = chunk["choices"][0]["delta"].get("content")
                if text:
                    yield text
