from typing import Dict, Iterator

from typing_extensions import override

from protocol.commands.base import FinalCommand


class EndDialog(FinalCommand):

    @staticmethod
    def token() -> str:
        return "end-dialog"


