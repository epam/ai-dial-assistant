from typing import Any

from langchain_community.tools.openapi.utils.api_models import (
    APIOperation,
    APIPropertyBase,
)
from typing_extensions import override

from aidial_assistant.chain.command_chain import CommandDict
from aidial_assistant.commands.base import (
    Command,
    ExecutionCallback,
    ResultObject,
    TextResult,
)
from aidial_assistant.commands.open_api import OpenAPIChatCommand
from aidial_assistant.commands.plugin_callback import PluginChainCallback
from aidial_assistant.commands.run_plugin import PluginInfo
from aidial_assistant.model.model_client import (
    Message,
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.open_api.operation_selector import collect_operations
from aidial_assistant.tools_chain.tools_chain import ToolsChain
from aidial_assistant.utils.open_ai import Tool, construct_function


def _construct_property(p: APIPropertyBase) -> dict[str, Any]:
    parameter = {
        "type": p.type,
        "description": p.description,
        "default": p.default,
    }
    return {k: v for k, v in parameter.items() if v is not None}


def _construct_function(op: APIOperation) -> Tool:
    properties = {}
    required = []
    for p in op.properties:
        properties[p.name] = _construct_property(p)

        if p.required:
            required.append(p.name)

    if op.request_body is not None:
        for p in op.request_body.properties:
            properties[p.name] = _construct_property(p)

            if p.required:
                required.append(p.name)

    return construct_function(
        op.operation_id, op.description or "", properties, required
    )


class RunTool(Command):
    def __init__(self, model: ModelClient, addon: PluginInfo):
        self.model = model
        self.addon = addon

    @staticmethod
    def token():
        return "run-tool"

    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        if "query" not in args:
            raise Exception("query is required")

        query = args["query"]

        ops = collect_operations(
            self.addon.info.open_api, self.addon.info.ai_plugin.api.url
        )
        tools: list[Tool] = [_construct_function(op) for op in ops.values()]

        def create_command(op: APIOperation):
            return lambda: OpenAPIChatCommand(op, self.addon.auth)

        command_dict: CommandDict = {
            name: create_command(op) for name, op in ops.items()
        }

        chain = ToolsChain(self.model, tools, command_dict)

        messages = [
            Message.system(self.addon.info.ai_plugin.description_for_model),
            Message.user(query),
        ]
        chain_callback = PluginChainCallback(execution_callback)
        try:
            await chain.run_chat(messages, chain_callback)
            return TextResult(chain_callback.result)
        except ReasonLengthException:
            return TextResult(chain_callback.result)
