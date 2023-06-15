from typing import Dict

from typing_extensions import override

from protocol.commands.base import Command
from utils.printing import print_red
from utils.text import indent


class SayOrAsk(Command):
    message: str

    @staticmethod
    def token():
        return "say-or-ask"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and len(dict["args"]) == 1
        self.message = dict["args"][0]

    @override
    def execute(self) -> str:
        print_red(indent(self.message, 0, ">"))
        return input()
