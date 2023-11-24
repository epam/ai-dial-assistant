from unittest.mock import Mock

import pytest
from jinja2 import Template

from aidial_assistant.chain.history import History, MessageScope, ScopedMessage
from aidial_assistant.model.model_client import Message, ModelClient

TRIMMING_TEST_DATA = [
    (0, [0, 1, 2, 3, 4, 5, 6]),
    (1, [0, 2, 3, 4, 5, 6]),
    (2, [0, 2, 6]),
    (3, [0, 2, 6]),
    (4, [0, 2, 6]),
]

MAX_PROMPT_TOKENS = 123


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "discarded_messages,expected_indices", TRIMMING_TEST_DATA
)
async def test_history_trimming(
    discarded_messages: int, expected_indices: list[int]
):
    history = History(
        assistant_system_message_template=Template(""),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(message=Message.system(content="a")),
            ScopedMessage(message=Message.user(content="b")),
            ScopedMessage(message=Message.system(content="c")),
            ScopedMessage(
                message=Message.assistant(content="d"),
                scope=MessageScope.INTERNAL,
            ),
            ScopedMessage(
                message=Message.user(content="e"), scope=MessageScope.INTERNAL
            ),
            ScopedMessage(message=Message.assistant(content="f")),
            ScopedMessage(message=Message.user(content="g")),
        ],
    )

    model_client = Mock(spec=ModelClient)
    model_client.get_discarded_messages.return_value = discarded_messages

    actual = await history.trim(MAX_PROMPT_TOKENS, model_client)

    assert (
        actual.assistant_system_message_template
        == history.assistant_system_message_template
    )
    assert actual.best_effort_template == history.best_effort_template
    assert actual.scoped_messages == [
        history.scoped_messages[i] for i in expected_indices
    ]


@pytest.mark.asyncio
async def test_trimming_overflow():
    history = History(
        assistant_system_message_template=Template(""),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(message=Message.system(content="a")),
            ScopedMessage(message=Message.user(content="b")),
        ],
    )

    model_client = Mock(spec=ModelClient)
    model_client.get_discarded_messages.return_value = 1

    with pytest.raises(Exception) as exc_info:
        await history.trim(MAX_PROMPT_TOKENS, model_client)

    assert (
        str(exc_info.value) == "No user messages left after history truncation."
    )


@pytest.mark.asyncio
async def test_trimming_with_incorrect_message_sequence():
    history = History(
        assistant_system_message_template=Template(""),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(
                message=Message.user(content="a"), scope=MessageScope.INTERNAL
            ),
            ScopedMessage(message=Message.user(content="b")),
        ],
    )

    model_client = Mock(spec=ModelClient)
    model_client.get_discarded_messages.return_value = 1

    with pytest.raises(Exception) as exc_info:
        await history.trim(MAX_PROMPT_TOKENS, model_client)

    assert (
        str(exc_info.value)
        == "Internal messages must be followed by an assistant reply."
    )
