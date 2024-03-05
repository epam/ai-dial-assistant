import json
from unittest.mock import MagicMock, Mock, call

import httpx
import pytest
from jinja2 import Template
from openai import BadRequestError

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.chain.command_chain import (
    CommandChain,
    LimitExceededException,
    ModelRequestLimiter,
)
from aidial_assistant.chain.history import History, ScopedMessage
from aidial_assistant.commands.base import Command, TextResult
from aidial_assistant.model.model_client import ModelClient
from aidial_assistant.utils.open_ai import (
    assistant_message,
    system_message,
    user_message,
)
from tests.utils.async_helper import to_async_string, to_async_strings

SYSTEM_MESSAGE = "<system message>"
USER_MESSAGE = "<user message>"
ENFORCE_JSON_FORMAT = "\n\n**Protocol reminder: reply with commands**"
BEST_EFFORT_ANSWER = "<best effort answer>"
NO_TOKENS_ERROR = "No tokens left"
FAILED_PROTOCOL_ERROR = "The next constructed API request is incorrect."
LIMIT_EXCEEDED_ERROR = "<limit exceeded error>"
TEST_COMMAND_NAME = "<test command>"
TEST_COMMAND_OUTPUT = "<test command result>"
TEST_COMMAND_REQUEST = json.dumps(
    {
        "commands": [
            {"command": TEST_COMMAND_NAME, "arguments": {"arg": "value"}}
        ]
    }
)
TEST_COMMAND_RESPONSE = json.dumps(
    {"responses": [{"status": "SUCCESS", "response": TEST_COMMAND_OUTPUT}]}
)
TEST_HISTORY = History(
    assistant_system_message_template=Template(
        "system_prefix={{system_prefix}}"
    ),
    best_effort_template=Template(
        "user_message={{message}}, error={{error}}, dialogue={{dialogue}}"
    ),
    scoped_messages=[
        ScopedMessage(message=system_message(SYSTEM_MESSAGE), user_index=0),
        ScopedMessage(message=user_message(USER_MESSAGE), user_index=1),
    ],
)


@pytest.mark.asyncio
async def test_model_doesnt_support_protocol():
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = to_async_strings(
        ["cannot reply in JSON format", BEST_EFFORT_ANSWER]
    )
    command_chain = CommandChain(
        name="TEST",
        model_client=model_client,
        command_dict={},
        max_retry_count=0,
    )
    chain_callback = Mock(spec=ChainCallback)
    result_callback = Mock(spec=ResultCallback)
    chain_callback.result_callback.return_value = result_callback

    await command_chain.run_chat(history=TEST_HISTORY, callback=chain_callback)

    assert chain_callback.on_error.call_args_list == [
        call("Error", "The model failed to construct addon request."),
    ]
    assert result_callback.on_result.call_args_list == [
        call(BEST_EFFORT_ANSWER)
    ]
    assert model_client.agenerate.call_args_list == [
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(f"{USER_MESSAGE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(SYSTEM_MESSAGE),
                user_message(USER_MESSAGE),
            ]
        ),
    ]


@pytest.mark.asyncio
async def test_model_partially_supports_protocol():
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = to_async_strings(
        [
            TEST_COMMAND_REQUEST,
            "cannot reply in JSON format anymore",
            BEST_EFFORT_ANSWER,
        ]
    )
    test_command = Mock(spec=Command)
    test_command.execute.return_value = TextResult(TEST_COMMAND_OUTPUT)
    command_chain = CommandChain(
        name="TEST",
        model_client=model_client,
        command_dict={TEST_COMMAND_NAME: lambda *_: test_command},
        max_retry_count=0,
    )
    chain_callback = MagicMock(spec=ChainCallback)
    result_callback = Mock(spec=ResultCallback)
    chain_callback.result_callback.return_value = result_callback
    succeeded_dialogue = [
        assistant_message(TEST_COMMAND_REQUEST),
        user_message(TEST_COMMAND_RESPONSE),
    ]

    await command_chain.run_chat(history=TEST_HISTORY, callback=chain_callback)

    assert chain_callback.on_error.call_args_list == [
        call("Error", "The model failed to construct addon request."),
    ]
    assert result_callback.on_result.call_args_list == [
        call(BEST_EFFORT_ANSWER)
    ]
    assert model_client.agenerate.call_args_list == [
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(f"{USER_MESSAGE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(USER_MESSAGE),
                assistant_message(TEST_COMMAND_REQUEST),
                user_message(f"{TEST_COMMAND_RESPONSE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(SYSTEM_MESSAGE),
                user_message(
                    f"user_message={USER_MESSAGE}, error={FAILED_PROTOCOL_ERROR}, dialogue={succeeded_dialogue}"
                ),
            ]
        ),
    ]


@pytest.mark.asyncio
async def test_no_tokens_for_tools():
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = [
        to_async_string(TEST_COMMAND_REQUEST),
        BadRequestError(
            message=NO_TOKENS_ERROR,
            response=httpx.Response(
                request=httpx.Request("GET", "http://localhost"),
                status_code=400,
            ),
            body={
                "type": "<error type>",
                "code": "<error code>",
                "param": "<param>",
            },
        ),
        to_async_string(BEST_EFFORT_ANSWER),
    ]
    test_command = Mock(spec=Command)
    test_command.execute.return_value = TextResult(TEST_COMMAND_OUTPUT)
    command_chain = CommandChain(
        name="TEST",
        model_client=model_client,
        command_dict={TEST_COMMAND_NAME: lambda *_: test_command},
        max_retry_count=0,
    )
    chain_callback = MagicMock(spec=ChainCallback)
    result_callback = Mock(spec=ResultCallback)
    chain_callback.result_callback.return_value = result_callback

    await command_chain.run_chat(history=TEST_HISTORY, callback=chain_callback)

    assert chain_callback.on_error.call_args_list == [
        call("Error", NO_TOKENS_ERROR)
    ]
    assert result_callback.on_result.call_args_list == [
        call(BEST_EFFORT_ANSWER)
    ]
    assert model_client.agenerate.call_args_list == [
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(f"{USER_MESSAGE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(USER_MESSAGE),
                assistant_message(TEST_COMMAND_REQUEST),
                user_message(f"{TEST_COMMAND_RESPONSE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(SYSTEM_MESSAGE),
                user_message(
                    f"user_message={USER_MESSAGE}, error={NO_TOKENS_ERROR}, dialogue=[]"
                ),
            ]
        ),
    ]


@pytest.mark.asyncio
async def test_model_request_limit_exceeded():
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = to_async_strings(
        [TEST_COMMAND_REQUEST, BEST_EFFORT_ANSWER]
    )
    test_command = Mock(spec=Command)
    test_command.execute.return_value = TextResult(TEST_COMMAND_OUTPUT)
    command_chain = CommandChain(
        name="TEST",
        model_client=model_client,
        command_dict={TEST_COMMAND_NAME: lambda *_: test_command},
        max_retry_count=0,
    )
    chain_callback = MagicMock(spec=ChainCallback)
    result_callback = Mock(spec=ResultCallback)
    chain_callback.result_callback.return_value = result_callback
    model_request_limiter = Mock(spec=ModelRequestLimiter)
    model_request_limiter.verify_limit.side_effect = [
        None,
        LimitExceededException(LIMIT_EXCEEDED_ERROR),
    ]

    await command_chain.run_chat(
        history=TEST_HISTORY,
        callback=chain_callback,
        model_request_limiter=model_request_limiter,
    )

    assert result_callback.on_result.call_args_list == [
        call(BEST_EFFORT_ANSWER)
    ]
    assert model_client.agenerate.call_args_list == [
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(f"{USER_MESSAGE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(SYSTEM_MESSAGE),
                user_message(
                    f"user_message={USER_MESSAGE}, error={LIMIT_EXCEEDED_ERROR}, dialogue=[]"
                ),
            ]
        ),
    ]
    assert model_request_limiter.verify_limit.call_args_list == [
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(f"{USER_MESSAGE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
        call(
            [
                system_message(f"system_prefix={SYSTEM_MESSAGE}"),
                user_message(USER_MESSAGE),
                assistant_message(TEST_COMMAND_REQUEST),
                user_message(f"{TEST_COMMAND_RESPONSE}{ENFORCE_JSON_FORMAT}"),
            ]
        ),
    ]
