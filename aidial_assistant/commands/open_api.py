from typing import Any

from langchain.tools.openapi.utils.api_models import APIOperation
from typing_extensions import override

from aidial_assistant.commands.base import (
    Command,
    ExecutionCallback,
    ResultObject,
)
from aidial_assistant.open_api.requester import (
    OpenAPIEndpointRequester,
    ParamMapping,
)


class OpenAPIChatCommand(Command):
    @staticmethod
    def token() -> str:
        return "open-api-chat-command"

    def __init__(self, requester: OpenAPIEndpointRequester):
        self.requester = requester

    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        return await self.requester.execute(args)

    @classmethod
    def create(
        cls, base_url: str, operation: APIOperation, auth: str | None
    ) -> "OpenAPIChatCommand":
        path = base_url.rstrip("/") + operation.path
        method = operation.method
        param_mapping = ParamMapping(
            query_params=operation.query_params,
            body_params=operation.body_params,
            path_params=operation.path_params,
        )
        return cls(OpenAPIEndpointRequester(path, method, param_mapping, auth))
