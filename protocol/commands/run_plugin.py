from pathlib import Path
from typing import Tuple, List
from urllib.parse import urlparse

from jinja2 import Template
from langchain.tools import APIOperation
from typing_extensions import override

from chains.command_chain import ChainCallback, CommandChain, CommandCallback, ArgsCallback, \
    ArgCallback, ResultCallback
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
from protocol.execution_context import CommandDict, ExecutionContext
from utils.open_ai_plugin import get_open_ai_plugin_info
from utils.printing import print_exception


def get_base_url(url: str) -> str:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    return base_url


class PluginArgCallback(ArgCallback):
    def __init__(self, arg_index: int, callback: ExecutionCallback):
        self.arg_index = arg_index
        self.callback = callback

    @override
    def on_arg_start(self):
        self.callback.on_message('"' if self.arg_index == 0 else ', "')

    @override
    def on_arg(self, token: str):
        self.callback.on_message(token.replace('"', '\\"'))

    @override
    def on_arg_end(self):
        self.callback.on_message('"')


class PluginArgsCallback(ArgsCallback):
    def __init__(self, callback: ExecutionCallback):
        self.execution_callback = callback
        self.arg_index = -1

    @override
    def on_args_start(self):
        self.execution_callback.on_message("(")

    @override
    def arg_callback(self) -> ArgCallback:
        self.arg_index += 1
        return PluginArgCallback(self.arg_index, self.execution_callback)

    @override
    def on_args_end(self):
        self.execution_callback.on_message(")\n")


class PluginCommandCallback(CommandCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    @override
    def on_command(self, command: str):
        self.callback.on_message(f">>> {command}")

    @override
    def args_callback(self) -> ArgsCallback:
        return PluginArgsCallback(self.callback)

    @override
    def execution_callback(self) -> ExecutionCallback:
        return self.callback

    @override
    def on_result(self, response):
        self.callback.on_message(f"<<< {response}\n")


class PluginResultCallback(ResultCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    def on_start(self):
        pass

    def on_result(self, token):
        self.callback.on_message(token)

    def on_end(self):
        self.callback.on_message("\n")


class PluginChainCallback(ChainCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    @override
    def on_start(self):
        pass

    @override
    def command_callback(self) -> PluginCommandCallback:
        return PluginCommandCallback(self.callback)

    @override
    def on_end(self, error: str | None = None):
        pass

    @override
    def on_ai_message(self, message: str):
        pass

    @override
    def on_human_message(self, message: str):
        pass

    @override
    def result_callback(self) -> ResultCallback:
        return PluginResultCallback(self.callback)


class RunPlugin(Command):
    @staticmethod
    def token():
        return "run-plugin"

    @override
    def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
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
            system_prefix, commands = self._process_plugin_tool(conf, plugin)
            return self._run_plugin(name, query, system_prefix, commands, execution_callback)

        if isinstance(plugin, PluginOpenAI):
            # 1. Using plugin prompt approach + abbreviated endpoints
            system_prefix, commands = RunPlugin._process_plugin_open_ai_typescript_commands(
                plugin
            )
            return self._run_plugin(name, query, system_prefix, commands, execution_callback)

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


    def _process_plugin_tool(
        self, conf: Conf, plugin: PluginTool
    ) -> Tuple[str, dict[str, CommandConf]]:
        commands: dict[str, CommandConf] = {}

        for name in plugin.commands:
            if name not in conf.commands:
                raise ValueError(
                    f"Unknown command: {name}. Available commands: {conf.commands.keys()}"
                )
            commands[name] = conf.commands[name]

        system_prefix = plugin.system_prefix
        return system_prefix, commands

    def _run_plugin(
            self,
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
            return chat.run_chat(init_messages, PluginChainCallback(execution_callback))
        except Exception as e:
            print_exception()
            return "ERROR: " + str(e)
