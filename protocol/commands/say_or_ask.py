from typing import List

from typing_extensions import override

from protocol.commands.base import Command, ExecutionCallback, ResultObject, TextResult
from utils.printing import print_red
from utils.text import indent


class SayOrAsk(Command):
    @staticmethod
    def token():
        return "say-or-ask"

    @override
    async def execute(self, args: List[str], execution_callback: ExecutionCallback) -> ResultObject:
        self.assert_arg_count(args, 1)
        message = args[0]

        print_red(indent(message, 0, ">"))
        return TextResult(input())
