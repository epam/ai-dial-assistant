from unittest.mock import Mock, call

import pytest

from aidial_assistant.application.addons_dialogue_limiter import (
    AddonsDialogueLimiter,
)
from aidial_assistant.chain.command_chain import LimitExceededException
from aidial_assistant.model.model_client import ModelClient
from aidial_assistant.utils.open_ai import (
    assistant_message,
    system_message,
    user_message,
)

MAX_TOKENS = 1


@pytest.mark.asyncio
async def test_dialogue_size_is_ok():
    model = Mock(spec=ModelClient)
    model.count_tokens.side_effect = [1, 2]

    limiter = AddonsDialogueLimiter(MAX_TOKENS, model)
    initial_messages = [system_message("a"), user_message("b")]
    dialogue_messages = [
        assistant_message("c"),
        user_message("d"),
    ]

    await limiter.verify_limit(initial_messages)
    await limiter.verify_limit(initial_messages + dialogue_messages)

    assert model.count_tokens.call_args_list == [
        call(initial_messages),
        call(initial_messages + dialogue_messages),
    ]


@pytest.mark.asyncio
async def test_dialogue_overflow():
    model = Mock(spec=ModelClient)
    model.count_tokens.side_effect = [1, 3]

    limiter = AddonsDialogueLimiter(MAX_TOKENS, model)
    initial_messages = [system_message("a"), user_message("b")]
    dialogue_messages = [
        assistant_message("c"),
        user_message("d"),
    ]

    await limiter.verify_limit(initial_messages)
    with pytest.raises(LimitExceededException) as exc_info:
        await limiter.verify_limit(initial_messages + dialogue_messages)

    assert (
        str(exc_info.value)
        == f"Addons dialogue limit exceeded. Max tokens: {MAX_TOKENS}, actual tokens: 2."
    )
