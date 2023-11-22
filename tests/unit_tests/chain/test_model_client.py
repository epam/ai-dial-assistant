from unittest import mock
from unittest.mock import Mock, call

import pytest

from aidial_assistant.chain.model_client import (
    ExtraResultsCallback,
    ModelClient,
)
from aidial_assistant.utils.text import join_string
from tests.utils.async_helper import to_async_list

API_METHOD = "openai.ChatCompletion.acreate"
MODEL_ARGS = {"model": "args"}
BUFFER_SIZE = 321


@mock.patch(API_METHOD)
@pytest.mark.asyncio
async def test_discarded_messages(api):
    model_client = ModelClient(MODEL_ARGS, BUFFER_SIZE)
    api.return_value = to_async_list(
        [
            {
                "choices": [{"delta": {"content": ""}}],
                "statistics": {"discarded_messages": 2},
            }
        ]
    )
    extra_results_callback = Mock(spec=ExtraResultsCallback)

    await join_string(model_client.agenerate([], extra_results_callback))

    assert extra_results_callback.set_discarded_messages.call_args_list == [
        call(2)
    ]
