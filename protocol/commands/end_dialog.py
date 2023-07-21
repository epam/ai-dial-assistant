from protocol.commands.base import FinalCommand


class Reply(FinalCommand):

    @staticmethod
    def token() -> str:
        return "reply"


