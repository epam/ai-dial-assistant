from typing import AsyncIterator
from unittest.mock import Mock

import pytest
from jinja2 import Template
from pydantic import BaseModel

from aidial_assistant.chain.history import History, MessageScope, ScopedMessage
from aidial_assistant.chain.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    ReasonLengthException,
)

TRIMMING_TEST_DATA = [
    (0, [0, 1, 2, 3, 4, 5, 6]),
    (1, [0, 2, 3, 4, 5, 6]),
    (2, [0, 2, 6]),
    (3, [0, 2, 6]),
    (4, [0, 2, 6]),
]

MAX_PROMPT_TOKENS = 123


class ModelSideEffect(BaseModel):
    discarded_messages: int

    async def agenerate(
        self, _, callback: ExtraResultsCallback, **kwargs
    ) -> AsyncIterator[str]:
        callback.on_discarded_messages(self.discarded_messages)
        yield "dummy"
        raise ReasonLengthException()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_count,expected_indices", TRIMMING_TEST_DATA)
async def test_history_trimming(
    message_count: int, expected_indices: list[int]
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

    side_effect = ModelSideEffect(discarded_messages=message_count)
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = side_effect.agenerate

    actual = await history.trim(MAX_PROMPT_TOKENS, model_client)

    assert (
        actual.assistant_system_message_template
        == history.assistant_system_message_template
    )
    assert actual.best_effort_template == history.best_effort_template
    assert actual.scoped_messages == [
        history.scoped_messages[i] for i in expected_indices
    ]
    assert (
        model_client.agenerate.call_args.kwargs["max_prompt_tokens"]
        == MAX_PROMPT_TOKENS
    )


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

    side_effect = ModelSideEffect(discarded_messages=1)
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = side_effect.agenerate

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

    side_effect = ModelSideEffect(discarded_messages=1)
    model_client = Mock(spec=ModelClient)
    model_client.agenerate.side_effect = side_effect.agenerate

    with pytest.raises(Exception) as exc_info:
        await history.trim(MAX_PROMPT_TOKENS, model_client)

    assert (
        str(exc_info.value)
        == "Internal messages must be followed by an assistant reply."
    )
