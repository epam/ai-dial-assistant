from aidial_sdk.chat_completion import CustomContent, Message, Role

from aidial_assistant.chain.history import MessageScope, ScopedMessage
from aidial_assistant.model.model_client import Message as ModelMessage
from aidial_assistant.utils.state import parse_history

FIRST_USER_MESSAGE = "<first user message>"
SECOND_USER_MESSAGE = "<first user message>"
FIRST_ASSISTANT_MESSAGE = "<first assistant message>"
SECOND_ASSISTANT_MESSAGE = "<second assistant message>"
FIRST_REQUEST = '{"commands": [{"command": "run-addon", "args": ["<addon1 name>", "<addon1 query>"]}]}'
FIRST_REQUEST_FIXED = '{"commands": [{"command": "<addon1 name>", "arguments": {"query": "<addon1 query>"}}]}'
SECOND_REQUEST = '{"commands": [{"command": "<addon2 name>", "arguments": {"query": "<addon2 query>"}}]}'
FIRST_RESPONSE = "<first response>"
SECOND_RESPONSE = "<second response>"


def test_parse_history():
    messages = [
        Message(role=Role.USER, content=FIRST_USER_MESSAGE),
        Message(
            role=Role.ASSISTANT,
            content=FIRST_ASSISTANT_MESSAGE,
            custom_content=CustomContent(
                state={
                    "invocations": [
                        {
                            "index": 1,
                            "request": SECOND_REQUEST,
                            "response": SECOND_RESPONSE,
                        },
                        {
                            "index": 0,
                            "request": FIRST_REQUEST,
                            "response": FIRST_RESPONSE,
                        },
                    ]
                }
            ),
        ),
        Message(role=Role.USER, content=SECOND_USER_MESSAGE),
        Message(role=Role.ASSISTANT, content=SECOND_ASSISTANT_MESSAGE),
    ]

    assert parse_history(messages) == [
        ScopedMessage(
            scope=MessageScope.USER,
            message=ModelMessage.user(FIRST_USER_MESSAGE),
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=ModelMessage.assistant(FIRST_REQUEST_FIXED),
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=ModelMessage.user(FIRST_RESPONSE),
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=ModelMessage.assistant(SECOND_REQUEST),
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=ModelMessage.user(content=SECOND_RESPONSE),
        ),
        ScopedMessage(
            scope=MessageScope.USER,
            message=ModelMessage.assistant(FIRST_ASSISTANT_MESSAGE),
        ),
        ScopedMessage(
            scope=MessageScope.USER,
            message=ModelMessage.user(SECOND_USER_MESSAGE),
        ),
        ScopedMessage(
            scope=MessageScope.USER,
            message=ModelMessage.assistant(SECOND_ASSISTANT_MESSAGE),
        ),
    ]
