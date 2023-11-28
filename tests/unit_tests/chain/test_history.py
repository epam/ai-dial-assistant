from jinja2 import Template

from aidial_assistant.chain.history import History, MessageScope, ScopedMessage
from aidial_assistant.chain.model_client import Message

SYSTEM_MESSAGE = "<system message>"
USER_MESSAGE = "<user message>"
ASSISTANT_MESSAGE = "<assistant message>"


def test_protocol_messages():
    history = History(
        assistant_system_message_template=Template(
            "system message={{system_prefix}}"
        ),
        best_effort_template=Template(""),
        scoped_messages=[
            ScopedMessage(
                scope=MessageScope.USER, message=Message.system(SYSTEM_MESSAGE)
            ),
            ScopedMessage(
                scope=MessageScope.USER, message=Message.user(USER_MESSAGE)
            ),
            ScopedMessage(
                scope=MessageScope.USER,
                message=Message.assistant(ASSISTANT_MESSAGE),
            ),
        ],
    )

    assert history.to_protocol_messages() == [
        Message.system(f"system message={SYSTEM_MESSAGE}"),
        Message.user(USER_MESSAGE),
        Message.assistant(
            f'{{"commands": [{{"command": "reply", "args": ["{ASSISTANT_MESSAGE}"]}}]}}'
        ),
    ]
