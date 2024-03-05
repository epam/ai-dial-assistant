from abc import ABC
from itertools import islice
from typing import Any, AsyncIterator, List

from aidial_sdk.utils.merge_chunks import merge
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)

from aidial_assistant.utils.open_ai import Usage


class ReasonLengthException(Exception):
    pass


class ExtraResultsCallback:
    def on_discarded_messages(self, discarded_messages: list[int]):
        pass

    def on_prompt_tokens(self, prompt_tokens: int):
        pass

    def on_tool_calls(
        self, tool_calls: list[ChatCompletionMessageToolCallParam]
    ):
        pass


async def _flush_stream(stream: AsyncIterator[str]):
    try:
        async for _ in stream:
            pass
    except ReasonLengthException:
        pass


def _discarded_messages_count_to_indices(
    messages: list[ChatCompletionMessageParam], discarded_messages: int
) -> list[int]:
    return list(
        islice(
            (
                i
                for i, message in enumerate(messages)
                if message["role"] != "system"
            ),
            discarded_messages,
        )
    )


class ModelClient(ABC):
    def __init__(self, client: AsyncOpenAI, model_args: dict[str, Any]):
        self.client = client
        self.model_args = model_args

        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    async def agenerate(
        self,
        messages: List[ChatCompletionMessageParam],
        extra_results_callback: ExtraResultsCallback | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        model_result = await self.client.chat.completions.create(
            **self.model_args,
            extra_body=kwargs,
            stream=True,
            messages=messages,
        )

        finish_reason_length = False
        tool_calls_chunks: list[list[dict[str, Any]]] = []
        async for chunk in model_result:
            chunk_dict = chunk.dict()
            usage: Usage | None = chunk_dict.get("usage")
            if usage:
                prompt_tokens = usage["prompt_tokens"]
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += usage["completion_tokens"]
                if extra_results_callback:
                    extra_results_callback.on_prompt_tokens(prompt_tokens)

            if extra_results_callback:
                discarded_messages: int | list[int] | None = chunk_dict.get(
                    "statistics", {}
                ).get("discarded_messages")
                if discarded_messages is not None:
                    extra_results_callback.on_discarded_messages(
                        _discarded_messages_count_to_indices(
                            messages, discarded_messages
                        )
                        if isinstance(discarded_messages, int)
                        else discarded_messages
                    )

            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                yield delta.content

            if delta.tool_calls:
                tool_calls_chunks.append(
                    [
                        tool_call_chunk.dict()
                        for tool_call_chunk in delta.tool_calls
                    ]
                )

            if choice.finish_reason == "length":
                finish_reason_length = True

        if finish_reason_length:
            raise ReasonLengthException()

        if extra_results_callback and tool_calls_chunks:
            tool_calls: list[ChatCompletionMessageToolCallParam] = [
                ChatCompletionMessageToolCallParam(**tool_call)
                for tool_call in merge(*tool_calls_chunks)
            ]
            extra_results_callback.on_tool_calls(tool_calls)

    # TODO: Use a dedicated endpoint for counting tokens.
    #  This request may throw an error if the number of tokens is too large.
    async def count_tokens(
        self, messages: list[ChatCompletionMessageParam]
    ) -> int:
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
    # https://github.com/epam/ai-dial-assistant/issues/39
    async def get_discarded_messages(
        self, messages: list[ChatCompletionMessageParam], max_prompt_tokens: int
    ) -> list[int]:
        class DiscardedMessagesCallback(ExtraResultsCallback):
            def __init__(self):
                self.discarded_messages: list[int] | None = None

            def on_discarded_messages(self, discarded_messages: list[int]):
                self.discarded_messages = discarded_messages

        callback = DiscardedMessagesCallback()
        await _flush_stream(
            self.agenerate(
                messages,
                extra_results_callback=callback,
                max_prompt_tokens=max_prompt_tokens,
                max_tokens=1,
            )
        )
        if callback.discarded_messages is None:
            raise Exception("Discarded messages were not provided.")

        return callback.discarded_messages

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens
