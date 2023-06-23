from typing import List

from typing_extensions import override

from chains.command_chain import ExecutionCallback
from protocol.commands.base import Command
from utils.printing import print_red
from utils.text import indent


class SayOrAsk(Command):
    @staticmethod
    def token():
        return "say-or-ask"

    @override
    def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
        assert len(args) == 1
        message = args[0]

        print_red(indent(message, 0, ">"))
        return input()
