from enum import Enum

from jinja2 import Template
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from aidial_assistant.chain.command_result import (
    CommandInvocation,
    commands_to_text,
)
from aidial_assistant.chain.dialogue import Dialogue
from aidial_assistant.commands.reply import Reply
from aidial_assistant.model.model_client import ModelClient
from aidial_assistant.utils.open_ai import assistant_message, system_message


class ContextLengthExceeded(Exception):
    pass


class MessageScope(str, Enum):
    INTERNAL = "internal"  # internal dialog with plugins/addons, not visible to the user on the top level
    USER = "user"  # top-level dialog with the user


class ScopedMessage(BaseModel):
    scope: MessageScope = MessageScope.USER
    message: ChatCompletionMessageParam


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

    def to_protocol_messages(self) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        for index, scoped_message in enumerate(self.scoped_messages):
            message = scoped_message.message
            scope = scoped_message.scope

            if index == 0:
                if message["role"] == "system":
                    messages.append(
                        system_message(
                            self.assistant_system_message_template.render(
                                system_prefix=message["content"]
                            )
                        )
                    )
                else:
                    messages.append(
                        system_message(
                            self.assistant_system_message_template.render()
                        )
                    )
                    messages.append(message)
            elif scope == MessageScope.USER and message["role"] == "assistant":
                # Clients see replies in plain text, but the model should understand how to reply appropriately.
                content = commands_to_text(
                    [
                        CommandInvocation(
                            command=Reply.token(),
                            arguments={"message": message.get("content", "")},
                        )
                    ]
                )
                messages.append(assistant_message(content))
            else:
                messages.append(message)

        return messages

    def to_user_messages(self) -> list[ChatCompletionMessageParam]:
        return [
            scoped_message.message
            for scoped_message in self.scoped_messages
            if scoped_message.scope == MessageScope.USER
        ]

    def to_best_effort_messages(
        self, error: str, dialogue: Dialogue
    ) -> list[ChatCompletionMessageParam]:
        messages = self.to_user_messages()

        last_message = messages[-1].copy()
        last_message["content"] = self.best_effort_template.render(
            message=last_message.get("content", ""),
            error=error,
            dialogue=dialogue.messages,
        )
        messages[-1] = last_message

        return messages

    async def truncate(
        self, max_prompt_tokens: int, model_client: ModelClient
    ) -> "History":
        discarded_messages = await model_client.get_discarded_messages(
            self.to_protocol_messages(),
            max_prompt_tokens,
        )

        if discarded_messages > 0:
            return History(
                assistant_system_message_template=self.assistant_system_message_template,
                best_effort_template=self.best_effort_template,
                scoped_messages=self._skip_messages(discarded_messages),
            )

        return self

    @property
    def user_message_count(self) -> int:
        return self._user_message_count

    def _skip_messages(self, discarded_messages: int) -> list[ScopedMessage]:
        messages: list[ScopedMessage] = []
        current_message = self.scoped_messages[0]
        message_iterator = iter(self.scoped_messages)
        for _ in range(discarded_messages):
            current_message = next(message_iterator)
            while current_message.message["role"] == "system":
                # System messages should be kept in the history
                messages.append(current_message)
                current_message = next(message_iterator)

        if current_message.scope == MessageScope.INTERNAL:
            while current_message.scope == MessageScope.INTERNAL:
                current_message = next(message_iterator)

            # Internal messages (i.e. addon requests/responses) are always followed by an assistant reply
            assert (
                current_message.message["role"] == "assistant"
            ), "Internal messages must be followed by an assistant reply."

        remaining_messages = list(message_iterator)
        assert (
            len(remaining_messages) > 0
        ), "No user messages left after history truncation."

        messages += remaining_messages

        return messages
