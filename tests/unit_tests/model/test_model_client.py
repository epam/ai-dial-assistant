from unittest import mock
from unittest.mock import Mock, call

import pytest

from aidial_assistant.model.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.utils.text import join_string
from tests.utils.async_helper import to_async_iterator

API_METHOD = "openai.ChatCompletion.acreate"
MODEL_ARGS = {"model": "args"}
BUFFER_SIZE = 321


@mock.patch(API_METHOD)
@pytest.mark.asyncio
async def test_discarded_messages(api):
    model_client = ModelClient(MODEL_ARGS, BUFFER_SIZE)
    api.return_value = to_async_iterator(
        [
            {
                "choices": [{"delta": {"content": ""}}],
                "statistics": {"discarded_messages": 2},
            }
        ]
    )
    extra_results_callback = Mock(spec=ExtraResultsCallback)

    await join_string(model_client.agenerate([], extra_results_callback))

    assert extra_results_callback.on_discarded_messages.call_args_list == [
        call(2)
    ]


@mock.patch(API_METHOD)
@pytest.mark.asyncio
async def test_content(api):
    model_client = ModelClient(MODEL_ARGS, BUFFER_SIZE)
    api.return_value = to_async_iterator(
        [
            {"choices": [{"delta": {"content": "one, "}}]},
            {"choices": [{"delta": {"content": "two, "}}]},
            {"choices": [{"delta": {"content": "three"}}]},
        ]
    )

    assert await join_string(model_client.agenerate([])) == "one, two, three"


@mock.patch(API_METHOD)
@pytest.mark.asyncio
async def test_reason_length_with_usage(api):
    model_client = ModelClient(MODEL_ARGS, BUFFER_SIZE)
    api.return_value = to_async_iterator(
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

    with pytest.raises(ReasonLengthException):
        async for chunk in model_client.agenerate([]):
            assert chunk == "text"

    assert model_client.total_prompt_tokens == 1
    assert model_client.total_completion_tokens == 2


@mock.patch(API_METHOD)
@pytest.mark.asyncio
async def test_api_args(api):
    model_client = ModelClient(MODEL_ARGS, BUFFER_SIZE)
    api.return_value = to_async_iterator([])
    messages = [
        Message.system(content="a"),
        Message.user(content="b"),
        Message.assistant(content="c"),
    ]

    await join_string(model_client.agenerate(messages))

    assert api.call_args_list == [
        call(
            messages=[
                {"role": "system", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            **MODEL_ARGS,
        )
    ]
