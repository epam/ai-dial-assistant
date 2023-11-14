from abc import ABC
from collections import defaultdict
from typing import Any, AsyncIterator, List

import openai
from aidial_sdk.chat_completion.request import Message
from aiohttp import ClientSession


class ReasonLengthException(Exception):
    pass


class UsagePublisher:
    def __init__(self):
        self.total_usage = defaultdict(int)

    def publish(self, usage: dict[str, int]):
        for k, v in usage.items():
            self.total_usage[k] += v

    @property
    def prompt_tokens(self) -> int:
        return self.total_usage["prompt_tokens"]

    @property
    def completion_tokens(self) -> int:
        return self.total_usage["completion_tokens"]


class ModelClient(ABC):
    def __init__(
        self,
        model_args: dict[str, Any],
        buffer_size: int,
    ):
        self.model_args = model_args
        self.buffer_size = buffer_size

    async def agenerate(
        self, messages: List[Message], usage_publisher: UsagePublisher
    ) -> AsyncIterator[str]:
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
                usage = chunk.get("usage")
                if usage:
                    usage_publisher.publish(usage)

                choice = chunk["choices"][0]
                text = choice["delta"].get("content")
                if text:
                    yield text

                if choice.get("finish_reason") == "length":
                    raise ReasonLengthException()
