from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, TypedDict

import openai
from aidial_sdk.chat_completion import Role
from aiohttp import ClientSession
from pydantic import BaseModel


class ReasonLengthException(Exception):
    pass


class Message(BaseModel):
    role: Role
    content: str

    def to_openai_message(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content):
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content):
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content):
        return cls(role=Role.ASSISTANT, content=content)


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class ExtraResultsCallback(ABC):
    @abstractmethod
    def on_discarded_messages(self, discarded_messages: int):
        pass


class ModelClient(ABC):
    def __init__(
        self,
        model_args: dict[str, Any],
        buffer_size: int,
    ):
        self.model_args = model_args
        self.buffer_size = buffer_size

        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    async def agenerate(
        self,
        messages: List[Message],
        extra_results_callback: ExtraResultsCallback | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        async with ClientSession(read_bufsize=self.buffer_size) as session:
            openai.aiosession.set(session)

            model_result = await openai.ChatCompletion.acreate(
                messages=[message.to_openai_message() for message in messages],
                **self.model_args | kwargs,
            )

            finish_reason_length = False
            async for chunk in model_result:  # type: ignore
                usage: Usage | None = chunk.get("usage")
                if usage:
                    self._prompt_tokens += usage["prompt_tokens"]
                    self._completion_tokens += usage["completion_tokens"]

                if extra_results_callback:
                    discarded_messages: int | None = chunk.get(
                        "statistics", {}
                    ).get("discarded_messages")
                    if discarded_messages is not None:
                        extra_results_callback.on_discarded_messages(
                            discarded_messages
                        )

                choice = chunk["choices"][0]
                text = choice["delta"].get("content")
                if text:
                    yield text

                if choice.get("finish_reason") == "length":
                    finish_reason_length = True

            if finish_reason_length:
                raise ReasonLengthException()

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens
