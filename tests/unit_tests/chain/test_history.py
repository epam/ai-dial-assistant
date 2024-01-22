from unittest.mock import Mock

import pytest
from jinja2 import Template

from aidial_assistant.chain.history import History, MessageScope, ScopedMessage
from aidial_assistant.model.model_client import ModelClient
from aidial_assistant.utils.open_ai import (
    assistant_message,
    system_message,
    user_message,
)

TRUNCATION_TEST_DATA = [
    ([], [0, 1, 2, 3, 4, 5, 6]),
    ([1], [0, 2, 3, 4, 5, 6]),
    ([1, 3], [0, 2, 6]),
    ([1, 3, 4], [0, 2, 6]),
    ([1, 3, 4, 5], [0, 2, 6]),
]

MAX_PROMPT_TOKENS = 123


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "discarded_model_messages,expected_indices", TRUNCATION_TEST_DATA
)
async def test_history_truncation(
    discarded_model_messages, expected_indices: list[int]
):
    full_history = History(
        assistant_system_message_template=Template(""),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(message=system_message("a"), user_index=0),
            ScopedMessage(message=user_message("b"), user_index=1),
            ScopedMessage(message=system_message("c"), user_index=2),
            ScopedMessage(
                message=assistant_message("d"),
                scope=MessageScope.INTERNAL,
                user_index=3,
            ),
            ScopedMessage(
                message=user_message(content="e"),
                scope=MessageScope.INTERNAL,
                user_index=3,
            ),
            ScopedMessage(message=assistant_message("f"), user_index=3),
            ScopedMessage(message=user_message("g"), user_index=4),
        ],
    )

    model_client = Mock(spec=ModelClient)
    model_client.get_discarded_messages.return_value = discarded_model_messages

    truncated_history, _ = await full_history.truncate(
        model_client, MAX_PROMPT_TOKENS
    )

    assert (
        full_history.assistant_system_message_template
        == full_history.assistant_system_message_template
    )
    assert (
        truncated_history.best_effort_template
        == full_history.best_effort_template
    )
    assert truncated_history.scoped_messages == [
        full_history.scoped_messages[i] for i in expected_indices
    ]


@pytest.mark.asyncio
async def test_truncation_overflow():
    history = History(
        assistant_system_message_template=Template(""),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(message=system_message("a"), user_index=0),
            ScopedMessage(message=user_message("b"), user_index=1),
        ],
    )

    model_client = Mock(spec=ModelClient)
    model_client.get_discarded_messages.return_value = 1

    with pytest.raises(Exception) as exc_info:
        await history.truncate(model_client, MAX_PROMPT_TOKENS)

    assert (
        str(exc_info.value) == "No user messages left after history truncation."
    )


@pytest.mark.asyncio
async def test_truncation_with_incorrect_message_sequence():
    history = History(
        assistant_system_message_template=Template(""),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(
                message=user_message("a"),
                scope=MessageScope.INTERNAL,
                user_index=0,
            ),
            ScopedMessage(message=user_message("b"), user_index=0),
        ],
    )

    model_client = Mock(spec=ModelClient)
    model_client.get_discarded_messages.return_value = 1

    with pytest.raises(Exception) as exc_info:
        await history.truncate(model_client, MAX_PROMPT_TOKENS)

    assert (
        str(exc_info.value)
        == "Internal messages must be followed by an assistant reply."
    )


def test_protocol_messages_with_system_message():
    system_content = "<system message>"
    user_content = "<user message>"
    assistant_content = "<assistant message>"
    history = History(
        assistant_system_message_template=Template(
            "system message={{system_prefix}}"
        ),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(message=system_message(system_content), user_index=0),
            ScopedMessage(message=user_message(user_content), user_index=1),
            ScopedMessage(
                message=assistant_message(assistant_content), user_index=2
            ),
        ],
    )

    assert history.to_protocol_messages() == [
        system_message(f"system message={system_content}"),
        user_message(user_content),
        assistant_message(
            f'{{"commands": [{{"command": "reply", "arguments": {{"message": "{assistant_content}"}}}}]}}'
        ),
    ]
