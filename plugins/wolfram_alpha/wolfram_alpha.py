from typing import Dict

import wolframalpha
from typing_extensions import override

from protocol.commands.base import Command

app_id = "8TJ67Y-PJ8R4338TQ"


class WolframAlpha(Command):
    request: str

    @staticmethod
    def token():
        return "wolfram-alpha"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 1
        self.request = dict["args"][0]

    @override
    def execute(self) -> str:
        client = wolframalpha.Client(app_id)
        resp = client.query(self.request)

        return str(resp)
