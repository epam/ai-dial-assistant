from typing import Dict, List

import requests
from typing_extensions import override

from chains.command_chain import ExecutionCallback
from protocol.commands.base import Command


class HttpRequest(Command):
    @staticmethod
    def token() -> str:
        return "http-request"

    @override
    def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
        assert len(args) == 2 or len(args) == 3
        method = args[0]
        url = args[1]
        json = args[2] if len(args) == 3 else None

        response = requests.request(method, url, json=json)
        if response.status_code == 200:
            result = response.json()
            return str(result)
        else:
            return f"Error: Unable to fetch data: {response}"
