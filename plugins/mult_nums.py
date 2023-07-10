import operator
from functools import reduce
from typing import Any, Dict, List

from typing_extensions import override

from protocol.commands.base import Command, ExecutionCallback


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

    @override
    async def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
        operands = list(map(parse_to_number, args))

        return str(reduce(operator.mul, operands, 1))
