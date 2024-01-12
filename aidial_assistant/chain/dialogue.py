from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from aidial_assistant.utils.open_ai import assistant_message, user_message


class DialogueTurn(BaseModel):
    assistant_message: str
    user_message: str


class Dialogue:
    def __init__(self):
        self.messages: list[ChatCompletionMessageParam] = []

    def append(self, dialogue_turn: DialogueTurn):
        self.messages.append(assistant_message(dialogue_turn.assistant_message))
        self.messages.append(user_message(dialogue_turn.user_message))

    def pop(self):
        self.messages.pop()
        self.messages.pop()

    def is_empty(self):
        return not self.messages

    def dialogue_turn_count(self):
        return len(self.messages) // 2
