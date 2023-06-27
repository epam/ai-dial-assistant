from typing import List, Any

from typing_extensions import override

from langchain.tools.openapi.utils.api_models import APIOperation

from open_api.requester import OpenAPIEndpointRequester
from protocol.commands.base import Command, ExecutionCallback


class OpenAPIChatCommand(Command):
    op: APIOperation

    @staticmethod
    def token() -> str:
        return "open-api-chat-command"

    def __init__(self, op: APIOperation):
        self.op = op

    @override
    async def execute(self, args: List[Any], execution_callback: ExecutionCallback) -> dict:
        assert len(args) == 1

        return OpenAPIEndpointRequester(self.op).execute(args[0])
