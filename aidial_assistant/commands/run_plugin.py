from typing import List

from aidial_sdk.chat_completion.request import Message, Role
from langchain.tools import APIOperation
from typing_extensions import override

from aidial_assistant.application.prompts import (
    PLUGIN_SYSTEM_DIALOG_MESSAGE,
    RESP_DIALOG_PROMPT,
)
from aidial_assistant.chain.command_chain import CommandChain
from aidial_assistant.chain.execution_context import ExecutionContext
from aidial_assistant.chain.model_client import ModelClient
from aidial_assistant.commands.base import (
    Command,
    CommandConstructor,
    ExecutionCallback,
    JsonResult,
    ResultObject,
)
from aidial_assistant.commands.open_api import OpenAPIChatCommand
from aidial_assistant.commands.plugin_callback import PluginChainCallback
from aidial_assistant.commands.reply import Reply
from aidial_assistant.open_api.operation_selector import collect_operations
from aidial_assistant.utils.open_ai_plugin import OpenAIPluginInfo


class RunPlugin(Command):
    def __init__(
        self, model_client: ModelClient, plugins: dict[str, OpenAIPluginInfo]
    ):
        self.model_client = model_client
        self.plugins = plugins

    @staticmethod
    def token():
        return "run-plugin"

    @override
    async def execute(
        self, args: List[str], execution_callback: ExecutionCallback
    ) -> ResultObject:
        self.assert_arg_count(args, 2)
        name = args[0]
        query = args[1]

        return await self._run_plugin(
            name,
            query,
            self.model_client,
            execution_callback,
        )

    async def _run_plugin(
        self,
        name: str,
        query: str,
        model_client: ModelClient,
        execution_callback: ExecutionCallback,
    ) -> ResultObject:
        if name not in self.plugins:
            raise ValueError(
                f"Unknown plugin: {name}. Available plugins: {[*self.plugins.keys()]}"
            )

        plugin = self.plugins[name]
        ops = collect_operations(plugin.open_api, plugin.ai_plugin.api.url)
        api_schema = "\n\n".join([op.to_typescript() for op in ops.values()])

        def create_command(op: APIOperation):
            return lambda: OpenAPIChatCommand(op, plugin.auth)

        command_dict: dict[str, CommandConstructor] = {}
        for name, op in ops.items():
            # The function is necessary to capture the current value of op.
            # Otherwise, only first op will be used for all commands
            command_dict[name] = create_command(op)

        command_dict[Reply.token()] = Reply

        init_messages = [
            Message(
                role=Role.SYSTEM,
                content=PLUGIN_SYSTEM_DIALOG_MESSAGE.render(
                    command_names=ops.keys(),
                    api_description=plugin.ai_plugin.description_for_model,
                    api_schema=api_schema,
                ),
            ),
            Message(role=Role.USER, content=query),
        ]

        chat = CommandChain(
            model_client=model_client,
            name="PLUGIN:" + name,
            resp_prompt=RESP_DIALOG_PROMPT,
            ctx=ExecutionContext(command_dict),
        )

        return JsonResult(
            await chat.run_chat(
                init_messages, PluginChainCallback(execution_callback.on_token)
            )
        )
