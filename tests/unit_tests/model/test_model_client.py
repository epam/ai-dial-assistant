from unittest.mock import Mock, call

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from aidial_assistant.model.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.utils.text import join_string
from tests.utils.async_helper import to_async_iterator

MODEL_ARGS = {"model": "args"}


@pytest.mark.asyncio
async def test_discarded_messages():
    openai_client = Mock()
    openai_client.chat.completions.create.return_value = to_async_iterator(
        [
            ChatCompletionChunk(
                choices=[{"delta": {"content": ""}}],
                statistics={"discarded_messages": 2},
            )
        ]
    )
    model_client = ModelClient(openai_client, MODEL_ARGS)
    extra_results_callback = Mock(spec=ExtraResultsCallback)

    await join_string(model_client.agenerate([], extra_results_callback))

    assert extra_results_callback.on_discarded_messages.call_args_list == [
        call(2)
    ]


@pytest.mark.asyncio
async def test_content():
    openai_client = Mock(spec=AsyncOpenAI)
    openai_client.chat.completions.create.return_value = to_async_iterator(
        [
            {"choices": [{"delta": {"content": "one, "}}]},
            {"choices": [{"delta": {"content": "two, "}}]},
            {"choices": [{"delta": {"content": "three"}}]},
        ]
    )
    model_client = ModelClient(openai_client, MODEL_ARGS)

    assert await join_string(model_client.agenerate([])) == "one, two, three"


@pytest.mark.asyncio
async def test_reason_length_with_usage():
    openai_client = Mock(spec=AsyncOpenAI)
    openai_client.chat.completions.create.return_value = to_async_iterator(
        [
            {"choices": [{"delta": {"content": "text"}}]},
            {
                "choices": [
                    {"delta": {"content": ""}, "finish_reason": "length"}  # type: ignore
                ]
            },
            {
                "choices": [{"delta": {"content": ""}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        ]
    )
    model_client = ModelClient(openai_client, MODEL_ARGS)

    with pytest.raises(ReasonLengthException):
        async for chunk in model_client.agenerate([]):
            assert chunk == "text"

    assert model_client.total_prompt_tokens == 1
    assert model_client.total_completion_tokens == 2


@pytest.mark.asyncio
async def test_api_args():
    openai_client = Mock(spec=AsyncOpenAI)
    openai_client.chat.completions.create.return_value = to_async_iterator([])
    model_client = ModelClient(openai_client, MODEL_ARGS)
    messages = [
        Message.system(content="a"),
        Message.user(content="b"),
        Message.assistant(content="c"),
    ]

    await join_string(model_client.agenerate(messages))

    assert openai_client.chat.completions.create.call_args_list == [
        call(
            messages=[
                {"role": "system", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            **MODEL_ARGS,
        )
    ]
