from typing import Dict

from typing_extensions import override

from protocol.commands.base import Command


class EndDialog(Command):
    response: str

    @staticmethod
    def token() -> str:
        return "end-dialog"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 1
        self.response = dict["args"][0]

    @override
    def execute(self) -> str:
        pass
