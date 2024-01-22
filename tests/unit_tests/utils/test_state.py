from aidial_sdk.chat_completion import CustomContent, Message, Role

from aidial_assistant.application.request_data import _parse_history
from aidial_assistant.chain.history import MessageScope, ScopedMessage
from aidial_assistant.utils.open_ai import assistant_message, user_message

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

    assert _parse_history(messages) == [
        ScopedMessage(
            scope=MessageScope.USER,
            message=user_message(FIRST_USER_MESSAGE),
            user_index=0,
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=assistant_message(FIRST_REQUEST_FIXED),
            user_index=0,
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=user_message(FIRST_RESPONSE),
            user_index=0,
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=assistant_message(SECOND_REQUEST),
            user_index=0,
        ),
        ScopedMessage(
            scope=MessageScope.INTERNAL,
            message=user_message(content=SECOND_RESPONSE),
            user_index=0,
        ),
        ScopedMessage(
            scope=MessageScope.USER,
            message=assistant_message(FIRST_ASSISTANT_MESSAGE),
            user_index=1,
        ),
        ScopedMessage(
            scope=MessageScope.USER,
            message=user_message(SECOND_USER_MESSAGE),
            user_index=2,
        ),
        ScopedMessage(
            scope=MessageScope.USER,
            message=assistant_message(SECOND_ASSISTANT_MESSAGE),
            user_index=3,
        ),
    ]
