from pydantic import BaseModel

from aidial_assistant.model.model_client import Message


class Exchange(BaseModel):
    assistant_message: str
    user_message: str


class Dialogue:
    def __init__(self):
        self.messages: list[Message] = []

    def append(self, exchange: Exchange):
        self.messages.append(Message.assistant(exchange.assistant_message))
        self.messages.append(Message.user(exchange.user_message))

    def pop(self):
        self.messages.pop()
        self.messages.pop()

    def is_empty(self):
        return not self.messages

    def exchange_count(self):
        return len(self.messages) // 2
