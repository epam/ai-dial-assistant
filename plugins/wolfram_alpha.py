from typing import Dict, Iterator, List

import wolframalpha
from typing_extensions import override

from chains.command_chain import ExecutionCallback
from protocol.commands.base import Command

app_id = "8TJ67Y-PJ8R4338TQ"


class WolframAlpha(Command):
    @staticmethod
    def token():
        return "wolfram-alpha"

    @override
    def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
        assert len(args) == 1
        request = args[0]

        client = wolframalpha.Client(app_id)
        resp = client.query(request)

        return str(resp)
