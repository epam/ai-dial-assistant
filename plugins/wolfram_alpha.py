from typing import Dict, Iterator, List

import wolframalpha
from typing_extensions import override

from protocol.commands.base import Command, ExecutionCallback, ResultObject, TextResult

app_id = "8TJ67Y-PJ8R4338TQ"


class WolframAlpha(Command):
    @staticmethod
    def token():
        return "wolfram-alpha"

    @override
    async def execute(self, args: List[str], execution_callback: ExecutionCallback) -> ResultObject:
        assert len(args) == 1
        request = args[0]

        client = wolframalpha.Client(app_id)
        resp = client.query(request)

        return TextResult(str(resp))
