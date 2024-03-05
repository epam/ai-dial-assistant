from enum import Enum
from typing import Tuple, cast

from jinja2 import Template
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
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
    user_index: int


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

    def to_protocol_messages(self) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        scoped_message_iterator = iter(self.scoped_messages)
        if self._is_first_system_message():
            message = cast(
                ChatCompletionSystemMessageParam,
                next(scoped_message_iterator).message,
            )
            messages.append(
                system_message(
                    self.assistant_system_message_template.render(
                        system_prefix=message["content"]
                    )
                )
            )
        else:
            messages.append(
                system_message(self.assistant_system_message_template.render())
            )

        for scoped_message in scoped_message_iterator:
            message = scoped_message.message
            scope = scoped_message.scope

            if scope == MessageScope.USER and message["role"] == "assistant":
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
        self, model_client: ModelClient, max_prompt_tokens: int
    ) -> Tuple["History", list[int]]:
        discarded_messages = await self._get_discarded_messages(
            model_client, max_prompt_tokens
        )

        if not discarded_messages:
            return self, []

        discarded_messages_set = set(discarded_messages)
        return (
            History(
                assistant_system_message_template=self.assistant_system_message_template,
                best_effort_template=self.best_effort_template,
                scoped_messages=[
                    scoped_message
                    for index, scoped_message in enumerate(self.scoped_messages)
                    if index not in discarded_messages_set
                ],
            ),
            discarded_messages,
        )

    async def _get_discarded_messages(
        self, model_client: ModelClient, max_prompt_tokens: int
    ) -> list[int]:
        discarded_protocol_messages = await model_client.get_discarded_messages(
            self.to_protocol_messages(),
            max_prompt_tokens,
        )

        if discarded_protocol_messages:
            discarded_protocol_messages.sort()
            discarded_messages = (
                discarded_protocol_messages
                if self._is_first_system_message()
                else [index - 1 for index in discarded_protocol_messages]
            )
            user_indices = set(
                self.scoped_messages[index].user_index
                for index in discarded_messages
            )

            return [
                index
                for index, scoped_message in enumerate(self.scoped_messages)
                if scoped_message.user_index in user_indices
            ]

        return discarded_protocol_messages

    def _is_first_system_message(self) -> bool:
        return (
            len(self.scoped_messages) > 0
            and self.scoped_messages[0].message["role"] == "system"
        )
