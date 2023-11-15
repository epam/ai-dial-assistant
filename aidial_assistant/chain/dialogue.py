from aidial_assistant.chain.model_client import Message


class Dialogue:
    def __init__(self):
        self.messages: list[Message] = []

    def append(self, assistant_message: str, user_message: str):
        self.messages.append(Message.assistant(assistant_message))
        self.messages.append(Message.user(user_message))

    def pop(self):
        self.messages.pop()
        self.messages.pop()

    def is_empty(self):
        return not self.messages
