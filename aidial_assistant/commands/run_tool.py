from typing import Any

from langchain_community.tools.openapi.utils.api_models import APIOperation
from typing_extensions import override

from aidial_assistant.commands.base import (
    Command,
    CommandConstructor,
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
from aidial_assistant.tools_chain.tools_chain import CommandToolDict, ToolsChain
from aidial_assistant.utils.open_ai import system_message, user_message
from aidial_assistant.utils.open_api import construct_tool_from_spec


class RunTool(Command):
    def __init__(
        self, model: ModelClient, plugin: PluginInfo, max_completion_tokens: int
    ):
        self.model = model
        self.plugin = plugin
        self.max_completion_tokens = max_completion_tokens

    @staticmethod
    def token():
        return "run-tool"

    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        query = get_required_field(args, "query")

        spec = self.plugin.info.open_api
        spec_url = self.plugin.info.get_full_spec_url()

        def create_command(operation: APIOperation) -> CommandConstructor:
            # The function is necessary to capture the current value of op.
            # Otherwise, only first op will be used for all commands
            return lambda: OpenAPIChatCommand.create(
                spec_url, operation, self.plugin.auth
            )

        commands: CommandToolDict = {
            operation.operation_id: (create_command(operation), tool)
            for path in spec.paths
            for operation, tool in (
                (
                    APIOperation.from_openapi_spec(spec, path, method),
                    construct_tool_from_spec(spec, path, method),
                )
                for method in spec.get_methods_for_path(path)
            )
        }

        chain = ToolsChain(self.model, commands, self.max_completion_tokens)

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
