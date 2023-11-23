import json
from enum import Enum

from aidial_sdk.chat_completion import Role
from jinja2 import Template
from pydantic import BaseModel
from typing_extensions import override

from aidial_assistant.chain.dialogue import Dialogue
from aidial_assistant.chain.model_client import (
    ExtraResultsCallback,
    Message,
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.commands.reply import Reply


class ContextLengthExceeded(Exception):
    pass


class MessageScope(str, Enum):
    INTERNAL = "internal"  # internal dialog with plugins/addons, not visible to the user on the top level
    USER = "user"  # top-level dialog with the user


class ScopedMessage(BaseModel):
    scope: MessageScope = MessageScope.USER
    message: Message


class ModelExtraResultsCallback(ExtraResultsCallback):
    def __init__(self):
        self._discarded_messages: int | None = None

    @override
    def on_discarded_messages(self, discarded_messages: int):
        self._discarded_messages = discarded_messages

    @property
    def discarded_messages(self) -> int | None:
        return self._discarded_messages


class History:
    def __init__(
        self,
        assistant_system_message_template: Template,
        best_effort_template: Template,
        scoped_messages: list[ScopedMessage],
    ):
        self.assistant_system_message_template = (
            assistant_system_message_template
        )
        self.best_effort_template = best_effort_template
        self.scoped_messages = scoped_messages
        self._user_message_count = sum(
            1
            for message in scoped_messages
            if message.scope == MessageScope.USER
        )

    def to_protocol_messages(self) -> list[Message]:
        messages: list[Message] = []
        for scoped_message in self.scoped_messages:
            message = scoped_message.message
            if (
                scoped_message.scope == MessageScope.USER
                and message.role == Role.ASSISTANT
            ):
                # Clients see replies in plain text, but the model should understand how to reply appropriately.
                content = json.dumps(
                    {
                        "commands": {
                            "command": Reply.token(),
                            "args": [message.content],
                        }
                    }
                )
                messages.append(Message.assistant(content=content))
            else:
                messages.append(message)

        return messages

    def to_protocol_messages_with_system_message(self) -> list[Message]:
        messages = self.to_protocol_messages()
        if messages[0].role == Role.SYSTEM:
            messages[0] = Message.system(
                self.assistant_system_message_template.render(
                    system_prefix=messages[0].content
                )
            )
        else:
            messages.insert(
                0,
                Message.system(self.assistant_system_message_template.render()),
            )

        return messages

    def to_client_messages(self) -> list[Message]:
        return [
            scoped_message.message
            for scoped_message in self.scoped_messages
            if scoped_message.scope == MessageScope.USER
        ]

    def to_best_effort_messages(
        self, error: str, dialogue: Dialogue
    ) -> list[Message]:
        messages = self.to_client_messages()

        last_message = messages[-1]
        messages[-1] = Message(
            role=last_message.role,
            content=self.best_effort_template.render(
                message=last_message.content,
                error=error,
                dialogue=dialogue.messages,
            ),
        )

        return messages

    async def trim(
        self, max_prompt_tokens: int, model_client: ModelClient
    ) -> "History":
        extra_results_callback = ModelExtraResultsCallback()
        stream = model_client.agenerate(
            self.to_protocol_messages(),
            extra_results_callback,
            max_prompt_tokens=max_prompt_tokens,
            max_tokens=1,
        )
        try:
            async for _ in stream:
                pass
        except ReasonLengthException:
            # Expected for max_tokens=1
            pass

        if extra_results_callback.discarded_messages:
            return History(
                assistant_system_message_template=self.assistant_system_message_template,
                best_effort_template=self.best_effort_template,
                scoped_messages=self._skip_messages(
                    extra_results_callback.discarded_messages
                ),
            )

        return self

    def _skip_messages(self, message_count: int) -> list[ScopedMessage]:
        messages = []
        current_message = self.scoped_messages[0]
        message_iterator = iter(self.scoped_messages)
        for _ in range(message_count):
            current_message = next(message_iterator)
            while current_message.message.role == Role.SYSTEM:
                # System messages should be kept in the history
                messages.append(current_message)
                current_message = next(message_iterator)

        if current_message.scope == MessageScope.INTERNAL:
            while current_message.scope == MessageScope.INTERNAL:
                current_message = next(message_iterator)

            # Internal messages (i.e. addon requests/responses) are always followed by an assistant reply
            assert (
                current_message.message.role == Role.ASSISTANT
            ), "Internal messages must be followed by an assistant reply."

        remaining_messages = list(message_iterator)
        assert (
            len(remaining_messages) > 0
        ), "No user messages left after history truncation."

        messages += remaining_messages

        return messages

    def user_message_count(self) -> int:
        return self._user_message_count
