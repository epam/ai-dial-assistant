from aidial_assistant.commands.base import FinalCommand


class Reply(FinalCommand):
    @staticmethod
    def token() -> str:
        return "reply"
