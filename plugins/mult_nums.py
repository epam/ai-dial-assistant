import operator
from functools import reduce
from typing import Dict, List

from typing_extensions import override

from protocol.commands.base import Command


class BigNumberMultiplication(Command):
    args: List[int]

    @staticmethod
    def token():
        return "big-number-multiplication"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        self.args = dict["args"]

    @override
    def execute(self) -> int:
        return reduce(operator.mul, self.args, 1)
