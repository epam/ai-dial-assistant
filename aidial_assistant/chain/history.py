from enum import Enum

from aidial_sdk.chat_completion import Role
from jinja2 import Template
from pydantic import BaseModel

from aidial_assistant.chain.command_result import (
    CommandInvocation,
    commands_to_text,
)
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

    def to_protocol_messages(self) -> list[Message]:
        messages: list[Message] = []
        for index, scoped_message in enumerate(self.scoped_messages):
            scope = scoped_message.scope
            message = scoped_message.message
            if index == 0:
                messages.append(
                    Message.system(
                        self.assistant_system_message_template.render(
                            system_prefix=message.content
                            if message.role == Role.SYSTEM
                            else ""
                        )
                    )
                )

                if message.role != Role.SYSTEM:
                    messages.append(message)

            elif scope == MessageScope.USER and message.role == Role.ASSISTANT:
                # Clients see replies in plain text, but the model should understand how to reply appropriately.
                content = commands_to_text(
                    [
                        CommandInvocation(
                            command=Reply.token(), args=[message.content]
                        )
                    ]
                )
                messages.append(Message.assistant(content=content))
            else:
                messages.append(message)

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
