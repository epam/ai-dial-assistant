from typing import Dict, List

import requests
from typing_extensions import override

from protocol.commands.base import Command, ExecutionCallback, ResultObject, TextResult


class HttpRequest(Command):
    @staticmethod
    def token() -> str:
        return "http-request"

    @override
    async def execute(self, args: List[str], execution_callback: ExecutionCallback) -> ResultObject:
        assert len(args) == 2 or len(args) == 3
        method = args[0]
        url = args[1]
        json = args[2] if len(args) == 3 else None

        response = requests.request(method, url, json=json)
        if response.status_code == 200:
            result = response.json()
            return TextResult(str(result))
        else:
            return TextResult(f"Error: Unable to fetch data: {response}")
