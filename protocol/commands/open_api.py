from typing import Dict

from typing_extensions import override

from langchain.tools.openapi.utils.api_models import APIOperation
from open_api.requester import OpenAPIEndpointRequester
from protocol.commands.base import Command


class OpenAPIChatCommand(Command):
    op: APIOperation
    arg: dict

    @staticmethod
    def token() -> str:
        return "open-api-chat-command"

    def __init__(self, op: APIOperation, dict: Dict):
        self.dict = dict
        assert "args" in dict
        self.arg = dict["args"][0] if isinstance(dict["args"], list) else dict["args"]
        self.op = op

    @override
    def execute(self) -> dict:
        return OpenAPIEndpointRequester(self.op).execute(self.arg)
