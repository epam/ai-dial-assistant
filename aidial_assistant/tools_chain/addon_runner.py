from typing import Any

from langchain_community.tools.openapi.utils.api_models import (
    APIOperation,
    APIPropertyBase,
)

from aidial_assistant.commands.base import (
    ExecutionCallback,
    ResultObject,
    TextResult,
)
from aidial_assistant.commands.plugin_callback import PluginChainCallback
from aidial_assistant.commands.run_plugin import PluginInfo
from aidial_assistant.model.model_client import (
    ModelClient,
    Message,
    ReasonLengthException,
    Tool,
)
from aidial_assistant.open_api.operation_selector import collect_operations
from aidial_assistant.tools_chain.http_runner import HttpRunner
from aidial_assistant.tools_chain.tool_runner import ToolRunner
from aidial_assistant.tools_chain.tools_chain import ToolsChain


def build_property(p: APIPropertyBase) -> dict[str, Any]:
    parameter = {
        "type": p.type,
        "description": p.description,
        "default": p.default,
    }
    return {k: v for k, v in parameter.items() if v is not None}


def construct_function(op: APIOperation) -> Tool:
    properties = {}
    required = []
    for p in op.properties:
        properties[p.name] = build_property(p)

        if p.required:
            required.append(p.name)

    if op.request_body is not None:
        for p in op.request_body.properties:
            properties[p.name] = build_property(p)

            if p.required:
                required.append(p.name)

    return {
        "type": "function",
        "function": {
            "name": op.operation_id,
            "description": op.description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class AddonRunner(ToolRunner):
    def __init__(self, model: ModelClient, addons: dict[str, PluginInfo]):
        self.model = model
        self.addons = addons

    async def run(
        self,
        name: str,
        arg: dict[str, Any],
        execution_callback: ExecutionCallback,
    ) -> ResultObject:
        query: str = arg["query"]

        addon = self.addons[name]
        ops = collect_operations(
            addon.info.open_api, addon.info.ai_plugin.api.url
        )
        tools = [construct_function(op) for op in ops.values()]

        chain = ToolsChain(self.model, tools, HttpRunner(ops, addon.auth))

        messages = [
            Message.system(addon.info.ai_plugin.description_for_model),
            Message.user(query),
        ]
        chain_callback = PluginChainCallback(execution_callback)
        try:
            await chain.run_chat(messages, chain_callback)
            return TextResult(chain_callback.result)
        except ReasonLengthException:
            return TextResult(chain_callback.result)
