import operator
from functools import reduce
from typing import Any, Dict, List

from typing_extensions import override

from protocol.commands.base import Command


def parse_to_number(val: Any) -> int | float:
    if isinstance(val, (int, float)):
        return val
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            raise ValueError(f"Cannot parse '{val}' to number")


class BigNumberMultiplication(Command):
    args: List[float | int]

    @staticmethod
    def token():
        return "big-number-multiplication"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        self.args = list(map(parse_to_number, dict["args"]))

    @override
    def execute(self) -> Any:
        return reduce(operator.mul, self.args, 1)
