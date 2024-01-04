from abc import ABC
from typing import Any, AsyncIterator, List, TypedDict

from aidial_sdk.chat_completion import Role
from aidial_sdk.utils.merge_chunks import merge
from openai import AsyncOpenAI
from pydantic import BaseModel


class ReasonLengthException(Exception):
    pass


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_openai_message(self) -> dict[str, str]:
        result = {"role": self.role.value}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        return result

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


class Parameters(TypedDict):
    type: str
    properties: dict[str, Any]
    required: list[str]


class Function(TypedDict):
    name: str
    description: str
    parameters: Parameters


class Tool(TypedDict):
    type: str
    function: Function


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    index: int
    id: str
    type: str
    function: FunctionCall


class ExtraResultsCallback:
    def on_discarded_messages(self, discarded_messages: int):
        pass

    def on_prompt_tokens(self, prompt_tokens: int):
        pass

    def on_tool_calls(self, tool_calls: list[ToolCall]):
        pass


async def _flush_stream(stream: AsyncIterator[str]):
    try:
        async for _ in stream:
            pass
    except ReasonLengthException:
        pass


class ModelClient(ABC):
    def __init__(self, client: AsyncOpenAI, model_args: dict[str, Any]):
        self.client = client
        self.model_args = model_args

        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    async def agenerate(
        self,
        messages: List[Message],
        extra_results_callback: ExtraResultsCallback | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        model_result = await self.client.chat.completions.create(
            **self.model_args,
            extra_body=kwargs,
            stream=True,
            messages=[message.to_openai_message() for message in messages],
        )

        finish_reason_length = False
        tool_calls_chunks = []
        async for chunk in model_result:  # type: ignore
            chunk = chunk.dict()
            usage: Usage | None = chunk.get("usage")
            if usage:
                prompt_tokens = usage["prompt_tokens"]
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += usage["completion_tokens"]
                if extra_results_callback:
                    extra_results_callback.on_prompt_tokens(prompt_tokens)

            if extra_results_callback:
                discarded_messages: int | None = chunk.get(
                    "statistics", {}
                ).get("discarded_messages")
                if discarded_messages is not None:
                    extra_results_callback.on_discarded_messages(
                        discarded_messages
                    )

            choice = chunk["choices"][0]
            delta = choice["delta"]
            text = delta.get("content")
            if text:
                yield text

            tool_calls_chunk = delta.get("tool_calls")
            if tool_calls_chunk:
                tool_calls_chunks.append(tool_calls_chunk)

            if choice.get("finish_reason") == "length":
                finish_reason_length = True

        if finish_reason_length:
            raise ReasonLengthException()

        if extra_results_callback and tool_calls_chunks:
            tool_calls: list[ToolCall] = merge(*tool_calls_chunks)
            extra_results_callback.on_tool_calls(tool_calls)

    # TODO: Use a dedicated endpoint for counting tokens.
    #  This request may throw an error if the number of tokens is too large.
    async def count_tokens(self, messages: list[Message]) -> int:
        class PromptTokensCallback(ExtraResultsCallback):
            def __init__(self):
                self.token_count: int | None = None

            def on_prompt_tokens(self, prompt_tokens: int):
                self.token_count = prompt_tokens

        callback = PromptTokensCallback()
        await _flush_stream(
            self.agenerate(
                messages, extra_results_callback=callback, max_tokens=1
            )
        )
        if callback.token_count is None:
            raise Exception("No token count received.")

        return callback.token_count

    # TODO: Use a dedicated endpoint for discarded_messages.
    async def get_discarded_messages(
        self, messages: list[Message], max_prompt_tokens: int
    ) -> int:
        class DiscardedMessagesCallback(ExtraResultsCallback):
            def __init__(self):
                self.message_count: int | None = None

            def on_discarded_messages(self, discarded_messages: int):
                self.message_count = discarded_messages

        callback = DiscardedMessagesCallback()
        await _flush_stream(
            self.agenerate(
                messages,
                extra_results_callback=callback,
                max_prompt_tokens=max_prompt_tokens,
                max_tokens=1,
            )
        )
        if callback.message_count is None:
            raise Exception("No message count received.")

        return callback.message_count

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens
