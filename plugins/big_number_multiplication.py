from typing import Dict

from typing_extensions import override

from protocol.commands.base import Command


class BigNumberMultiplication(Command):
    a: int
    b: int

    @staticmethod
    def token():
        return "run-python"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 2
        self.a = dict["args"][0]
        self.b = dict["args"][1]

    @override
    def execute(self) -> str:
        return str(self.a * self.b)
