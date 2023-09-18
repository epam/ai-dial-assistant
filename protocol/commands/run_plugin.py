from typing import Tuple, List

from jinja2 import Template
from langchain.schema import HumanMessage
from langchain.tools import APIOperation
from typing_extensions import override

from chains.command_chain import CommandChain
from chains.model_client import ModelClient
from conf.project_conf import (
    CommandConf,
)
from open_api.operation_selector import (
    collect_operations,
)
from prompts.dialog import (
    PLUGIN_SYSTEM_DIALOG_MESSAGE,
    RESP_DIALOG_PROMPT,
    open_api_plugin_template,
)
from protocol.commands.base import Command, ExecutionCallback, ResultObject, JsonResult
from protocol.commands.end_dialog import Reply
from protocol.commands.open_api import OpenAPIChatCommand
from protocol.commands.plugin_callback import PluginChainCallback
from protocol.execution_context import CommandDict, ExecutionContext
from utils.open_ai_plugin import OpenAIPluginInfo


class RunPlugin(Command):
    def __init__(self, model_client: ModelClient, plugins: dict[str, OpenAIPluginInfo]):
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

        if name not in self.plugins:
            raise ValueError(
                f"Unknown plugin: {name}. Available plugins: {self.plugins.keys()}"
            )

        plugin = self.plugins[name]

        # 1. Using plugin prompt approach + abbreviated endpoints
        (
            system_prefix,
            commands,
        ) = await RunPlugin._process_plugin_open_ai_typescript_commands(plugin)
        return await RunPlugin._run_plugin(
            name, query, system_prefix, commands, self.model_client, execution_callback
        )

        # 2. Using custom prompt borrowed from LangChain
        # return self._process_plugin_open_ai_typescript(plugin)

        # 3. Using plugin prompt approach + full OpenAPI specification
        # system_prefix, commands = self._process_plugin_open_ai_json(conf, plugin)
        # return self._run_plugin(system_prefix, commands)

    @staticmethod
    async def _process_plugin_open_ai_typescript_commands(
        plugin: OpenAIPluginInfo,
    ) -> Tuple[str, dict[str, CommandConf]]:
        spec = plugin.open_api
        api_description = plugin.ai_plugin.description_for_model

        ops = collect_operations(spec, plugin.ai_plugin.api.url)
        api_schema = "\n\n".join([op.to_typescript() for op in ops.values()])

        system_prefix = Template(open_api_plugin_template).render(
            api_description=api_description, api_schema=api_schema
        )

        def create_command(op: APIOperation):
            return lambda: OpenAPIChatCommand(op, plugin.auth)

        commands: dict[str, CommandConf] = {}
        for name, op in ops.items():
            commands[name] = CommandConf(
                implementation=create_command(op),
                description="",
                args=["<JSON dict according to the API_SCHEMA>"],
                result="",
            )

        return system_prefix, commands

    @staticmethod
    async def _run_plugin(
        name: str,
        query: str,
        system_prefix: str,
        commands: dict[str, CommandConf],
        model_client: ModelClient,
        execution_callback: ExecutionCallback,
    ) -> ResultObject:
        command_dict: CommandDict = {Reply.token(): Reply}

        for name, command_spec in commands.items():
            command_dict[name] = command_spec.implementation

        init_messages = [
            PLUGIN_SYSTEM_DIALOG_MESSAGE.format(
                commands=commands, system_prefix=system_prefix
            ),
            HumanMessage(content=query),
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
