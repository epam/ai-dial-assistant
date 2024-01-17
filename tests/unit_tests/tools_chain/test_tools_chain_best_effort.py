import json

import pytest
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function

from aidial_assistant.tools_chain.tools_chain import ToolsChain
from aidial_assistant.utils.open_ai import (
    construct_tool,
    tool_calls_message,
    tool_message,
    user_message,
)
from tests.utils.mocks import (
    TestChainCallback,
    TestCommand,
    TestModelClient,
    TestModelRequestLimiter,
)

TEST_COMMAND_NAME = "<test command>"
TOOL_ID = "<tool id>"
TOOL_RESPONSE = "<tool response>"
BEST_EFFORT_RESPONSE = "<best effort response>"


@pytest.mark.asyncio
async def test_model_request_limit_exceeded():
    messages: list[ChatCompletionMessageParam] = [user_message("<query>")]
    command_args = json.dumps({"<test argument>": "<test value>"})
    tool_calls = [
        ChatCompletionMessageToolCallParam(
            id=TOOL_ID,
            function=Function(
                name=TEST_COMMAND_NAME,
                arguments=command_args,
            ),
            type="function",
        )
    ]
    tool = construct_tool(TEST_COMMAND_NAME, "", {}, [])
    model = TestModelClient(
        tool_calls={
            TestModelClient.agenerate_key(messages, tools=[tool]): tool_calls
        },
        results={TestModelClient.agenerate_key(messages): BEST_EFFORT_RESPONSE},
    )

    messages_with_dialogue = messages + [
        tool_calls_message(tool_calls=tool_calls),
        tool_message(TOOL_RESPONSE, TOOL_ID),
    ]
    model_request_limiter = TestModelRequestLimiter(messages_with_dialogue)
    callback = TestChainCallback()
    tools_chain = ToolsChain(
        model,
        commands={
            TEST_COMMAND_NAME: (
                lambda: TestCommand({command_args: TOOL_RESPONSE}),
                tool,
            )
        },
    )

    await tools_chain.run_chat(messages, callback, model_request_limiter)

    assert callback.mock_result_callback.result == BEST_EFFORT_RESPONSE
