from pydantic import BaseModel

from aidial_assistant.model.model_client import Message


class DialogueTurn(BaseModel):
    assistant_message: str
    user_message: str


class Dialogue:
    def __init__(self):
        self.messages: list[Message] = []

    def append(self, dialogue_turn: DialogueTurn):
        self.messages.append(Message.assistant(dialogue_turn.assistant_message))
        self.messages.append(Message.user(dialogue_turn.user_message))

    def pop(self):
        self.messages.pop()
        self.messages.pop()

    def is_empty(self):
        return not self.messages

    def dialogue_turn_count(self):
        return len(self.messages) // 2
