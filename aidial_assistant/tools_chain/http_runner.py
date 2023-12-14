from typing import Any

from langchain_community.tools.openapi.utils.api_models import APIOperation

from aidial_assistant.commands.base import ExecutionCallback, ResultObject
from aidial_assistant.open_api.requester import OpenAPIEndpointRequester
from aidial_assistant.tools_chain.tool_runner import ToolRunner


class HttpRunner(ToolRunner):
    def __init__(self, ops: dict[str, APIOperation], auth: str):
        self.ops = ops
        self.auth = auth

    async def run(
        self,
        name: str,
        arg: dict[str, Any],
        execution_callback: ExecutionCallback,
    ) -> ResultObject:
        return await OpenAPIEndpointRequester(
            self.ops[name], self.auth
        ).execute(arg)
