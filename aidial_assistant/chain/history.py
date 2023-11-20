import json
from enum import Enum

from aidial_sdk.chat_completion import Role
from jinja2 import Template
from pydantic import BaseModel

from aidial_assistant.chain.dialogue import Dialogue
from aidial_assistant.chain.model_client import Message
from aidial_assistant.commands.reply import Reply


class MessageScope(str, Enum):
    INTERNAL = "internal"  # internal dialog with plugins/addons, not visible to the user on the top level
    USER = "user"  # top-level dialog with the user


class ScopedMessage(BaseModel):
    scope: MessageScope = MessageScope.USER
    message: Message


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
        first_message = next(iter(messages))
        if first_message and first_message.role == Role.SYSTEM:
            messages[0] = Message.system(
                self.assistant_system_message_template.render(
                    system_prefix=first_message.content
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

    def trim(self, message_count: int) -> "History":
        if message_count >= len(self.scoped_messages):
            raise ValueError(
                f"Cannot trim {message_count} messages from history with {len(self.scoped_messages)} messages."
            )

        if message_count == 0:
            return self

        last_discarded_message = self.scoped_messages[message_count - 1]
        if last_discarded_message.scope == MessageScope.INTERNAL:
            while last_discarded_message.scope == MessageScope.INTERNAL:
                last_discarded_message = self.scoped_messages[message_count]
                message_count += 1

            assert last_discarded_message.message.role == Role.ASSISTANT

        return History(
            assistant_system_message_template=self.assistant_system_message_template,
            best_effort_template=self.best_effort_template,
            scoped_messages=self.scoped_messages[message_count:],
        )

    def user_message_count(self) -> int:
        return self._user_message_count
