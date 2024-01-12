from typing import Any

from langchain_community.tools.openapi.utils.api_models import (
    APIOperation,
    APIPropertyBase,
)
from openai.types.chat import ChatCompletionToolParam
from typing_extensions import override

from aidial_assistant.commands.base import (
    Command,
    ExecutionCallback,
    ResultObject,
    TextResult,
    get_required_field,
)
from aidial_assistant.commands.open_api import OpenAPIChatCommand
from aidial_assistant.commands.plugin_callback import PluginChainCallback
from aidial_assistant.commands.run_plugin import PluginInfo
from aidial_assistant.model.model_client import (
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.open_api.operation_selector import collect_operations
from aidial_assistant.tools_chain.tools_chain import (
    CommandTool,
    CommandToolDict,
    ToolsChain,
)
from aidial_assistant.utils.open_ai import (
    construct_tool,
    system_message,
    user_message,
)


def _construct_property(p: APIPropertyBase) -> dict[str, Any]:
    parameter = {
        "type": p.type,
        "description": p.description,
    }
    return {k: v for k, v in parameter.items() if v is not None}


def _construct_function(op: APIOperation) -> ChatCompletionToolParam:
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

    return construct_tool(
        op.operation_id, op.description or "", properties, required
    )


class RunTool(Command):
    def __init__(self, model: ModelClient, plugin: PluginInfo):
        self.model = model
        self.plugin = plugin

    @staticmethod
    def token():
        return "run-tool"

    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        query = get_required_field(args, "query")

        ops = collect_operations(
            self.plugin.info.open_api, self.plugin.info.ai_plugin.api.url
        )

        def create_command_tool(op: APIOperation) -> CommandTool:
            return lambda: OpenAPIChatCommand(
                op, self.plugin.auth
            ), _construct_function(op)

        command_tool_dict: CommandToolDict = {
            name: create_command_tool(op) for name, op in ops.items()
        }

        chain = ToolsChain(self.model, command_tool_dict)

        messages = [
            system_message(self.plugin.info.ai_plugin.description_for_model),
            user_message(query),
        ]
        chain_callback = PluginChainCallback(execution_callback)
        try:
            await chain.run_chat(messages, chain_callback)
        except ReasonLengthException:
            pass

        return TextResult(chain_callback.result)
