from typing import Dict

import requests
from typing_extensions import override

from protocol.commands.base import Command


class HttpRequest(Command):
    method: str
    url: str
    json: dict | None

    @staticmethod
    def token() -> str:
        return "http-request"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 2 or len(dict["args"]) == 3
        self.method = dict["args"][0]
        self.url = dict["args"][1]
        self.json = dict["args"][2] if len(dict["args"]) == 3 else None

    @override
    def execute(self) -> str:
        response = requests.request(self.method, self.url, json=self.json)
        if response.status_code == 200:
            result = response.json()
            return str(result)
        else:
            return f"Error: Unable to fetch data: {response}"
