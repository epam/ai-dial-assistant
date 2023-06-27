from pathlib import Path
from typing import Tuple, List
from urllib.parse import urlparse

from jinja2 import Template
from langchain.tools import APIOperation
from typing_extensions import override

from chains.callbacks.result_callback import ResultCallback
from chains.command_chain import CommandChain
from chains.callbacks.arg_callback import ArgCallback
from chains.callbacks.args_callback import ArgsCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.chain_callback import ChainCallback
from chains.model_client import ModelClient
from cli.main_args import parse_args
from conf.project_conf import (
    CommandConf,
    Conf,
    PluginCommand,
    PluginOpenAI,
    PluginTool,
    read_conf,
)
from llm.base import create_chat_from_conf
from open_api.operation_selector import (
    collect_operations,
)
from prompts.dialog import (
    PLUGIN_SYSTEM_DIALOG_MESSAGE,
    RESP_DIALOG_PROMPT,
    open_api_plugin_template,
)
from protocol.commands.base import Command, ExecutionCallback
from protocol.commands.end_dialog import EndDialog
from protocol.commands.open_api import OpenAPIChatCommand
from protocol.commands.plugin_callback import PluginChainCallback
from protocol.execution_context import CommandDict, ExecutionContext
from utils.open_ai_plugin import get_open_ai_plugin_info
from utils.printing import print_exception


def get_base_url(url: str) -> str:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    return base_url


class RunPlugin(Command):
    @staticmethod
    def token():
        return "run-plugin"

    @override
    async def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
        assert len(args) == 2
        name = args[0]
        query = args[1]

        conf = read_conf(Conf, Path("plugins") / "index.yaml")

        if name not in conf.plugins:
            raise ValueError(
                f"Unknown plugin: {name}. Available plugins: {conf.plugins.keys()}"
            )

        plugin = conf.plugins[name]

        if isinstance(plugin, PluginCommand):
            raise ValueError(f"Command isn't a plugin: {name}")

        if isinstance(plugin, PluginTool):
            system_prefix, commands = RunPlugin._process_plugin_tool(conf, plugin)
            return await RunPlugin._run_plugin(name, query, system_prefix, commands, execution_callback)

        if isinstance(plugin, PluginOpenAI):
            # 1. Using plugin prompt approach + abbreviated endpoints
            system_prefix, commands = RunPlugin._process_plugin_open_ai_typescript_commands(
                plugin
            )
            return await RunPlugin._run_plugin(name, query, system_prefix, commands, execution_callback)

            # 2. Using custom prompt borrowed from LangChain
            # return self._process_plugin_open_ai_typescript(plugin)

            # 3. Using plugin prompt approach + full OpenAPI specification
            # system_prefix, commands = self._process_plugin_open_ai_json(conf, plugin)
            # return self._run_plugin(system_prefix, commands)

        raise ValueError(f"Unknown plugin type: {plugin}")

    @staticmethod
    def _process_plugin_open_ai_typescript_commands(plugin: PluginOpenAI) -> Tuple[str, dict[str, CommandConf]]:
        info = get_open_ai_plugin_info(plugin.url)
        spec = info.open_api
        api_description = info.ai_plugin.description_for_model

        ops = collect_operations(spec)
        api_schema = "\n\n".join([op.to_typescript() for op in ops.values()])

        system_prefix = Template(open_api_plugin_template).render(
            api_description=api_description, api_schema=api_schema
        )

        def create_command(op: APIOperation):
            return lambda: OpenAPIChatCommand(op)

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
    def _process_plugin_tool(conf: Conf, plugin: PluginTool) -> Tuple[str, dict[str, CommandConf]]:
        commands: dict[str, CommandConf] = {}

        for name in plugin.commands:
            if name not in conf.commands:
                raise ValueError(
                    f"Unknown command: {name}. Available commands: {conf.commands.keys()}"
                )
            commands[name] = conf.commands[name]

        system_prefix = plugin.system_prefix
        return system_prefix, commands

    @staticmethod
    async def _run_plugin(
            name: str,
            query: str,
            system_prefix: str,
            commands: dict[str, CommandConf],
            execution_callback: ExecutionCallback,
    ) -> str:
        command_dict: CommandDict = {EndDialog.token(): EndDialog}

        for name, command_spec in commands.items():
            command_dict[name] = command_spec.implementation

        init_messages = [
            PLUGIN_SYSTEM_DIALOG_MESSAGE.format(
                commands=commands,
                system_prefix=system_prefix,
                query=query,
            )
        ]

        args = parse_args()
        model = create_chat_from_conf(args.openai_conf, args.chat_conf)

        chat = CommandChain(
            model_client=ModelClient(model=model),
            name="PLUGIN:" + name,
            resp_prompt=RESP_DIALOG_PROMPT,
            ctx=ExecutionContext(command_dict),
        )

        try:
            return await chat.run_chat(init_messages, PluginChainCallback(execution_callback))
        except Exception as e:
            print_exception()
            return "ERROR: " + str(e)
