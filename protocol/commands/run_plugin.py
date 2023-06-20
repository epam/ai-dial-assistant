from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

from jinja2 import Template
from langchain.tools import APIOperation
from typing_extensions import override

from chains.command_chain import ChainCallback, CommandChain
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
    OpenAPIClarification,
    collect_operations,
    select_open_api_operation,
)
from open_api.requester import OpenAPIEndpointRequester
from open_api.response_summarisarion import summarize_response
from plugins.http_request import HttpRequest
from prompts.dialog import (
    PLUGIN_SYSTEM_DIALOG_MESSAGE,
    RESP_DIALOG_PROMPT,
    open_ai_plugin_template,
    open_api_plugin_template,
)
from protocol.commands.base import Command
from protocol.commands.end_dialog import EndDialog
from protocol.commands.open_api import OpenAPIChatCommand
from protocol.execution_context import CommandDict, ExecutionContext
from utils.open_ai_plugin import get_open_ai_plugin_info
from utils.printing import print_exception


def get_base_url(url: str) -> str:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    return base_url


class RunPlugin(Command):
    callback: ChainCallback | None
    name: str
    query: str

    @staticmethod
    def token():
        return "run-plugin"

    def __init__(self, callback: ChainCallback | None, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 2
        self.name = dict["args"][0]
        self.query = dict["args"][1]
        self.callback = callback

    @override
    def execute(self) -> str:
        conf = read_conf(Conf, Path("plugins") / "index.yaml")

        if self.name not in conf.plugins:
            raise ValueError(
                f"Unknown plugin: {self.name}. Available plugins: {conf.plugins.keys()}"
            )

        plugin = conf.plugins[self.name]

        if isinstance(plugin, PluginCommand):
            raise ValueError(f"Command isn't a plugin: {self.name}")

        if isinstance(plugin, PluginTool):
            system_prefix, commands = self._process_plugin_tool(conf, plugin)
            return self._run_plugin(system_prefix, commands)

        elif isinstance(plugin, PluginOpenAI):
            # 1. Using plugin prompt approach + abbreviated endpoints
            system_prefix, commands = self._process_plugin_open_ai_typescript_commands(
                plugin
            )
            return self._run_plugin(system_prefix, commands)

            # 2. Using custom prompt borrowed from LangChain
            # return self._process_plugin_open_ai_typescript(plugin)

            # 3. Using plugin prompt approach + full OpenAPI specification
            # system_prefix, commands = self._process_plugin_open_ai_json(conf, plugin)
            # return self._run_plugin(system_prefix, commands)

        raise ValueError(f"Unknown plugin type: {plugin}")

    def _process_plugin_open_ai_typescript_commands(
        self, plugin: PluginOpenAI
    ) -> Tuple[str, dict[str, CommandConf]]:
        info = get_open_ai_plugin_info(plugin.url)
        spec = info.open_api
        api_description = info.ai_plugin.description_for_model

        ops = collect_operations(spec)
        api_schema = "\n\n".join([op.to_typescript() for op in ops.values()])

        system_prefix = Template(open_api_plugin_template).render(
            api_description=api_description, api_schema=api_schema
        )

        def create_command(op: APIOperation):
            return lambda d: OpenAPIChatCommand(op, d)

        commands: dict[str, CommandConf] = {}
        for name, op in ops.items():
            commands[name] = CommandConf(
                implementation=create_command(op),
                description="",
                args=["<JSON dict according to the API_SCHEMA>"],
                result="",
            )

        return system_prefix, commands

    def _process_plugin_open_ai_typescript(self, plugin: PluginOpenAI) -> str:
        info = get_open_ai_plugin_info(plugin.url)
        spec = info.open_api
        api_description = info.ai_plugin.description_for_model

        ops = collect_operations(spec)
        resp = select_open_api_operation(api_description, ops, self.query)
        if isinstance(resp, OpenAPIClarification):
            return resp.user_question

        api_response = OpenAPIEndpointRequester(ops[resp.command]).execute(resp.args)
        summary = summarize_response(api_response, self.query)
        return summary

    def _process_plugin_open_ai_json(
        self, conf: Conf, plugin: PluginOpenAI
    ) -> Tuple[str, dict[str, CommandConf]]:
        name = HttpRequest.token()
        commands: dict[str, CommandConf] = {name: conf.commands[name]}

        info = get_open_ai_plugin_info(plugin.url)

        open_api = info.open_api.json()
        description_for_model = info.ai_plugin.description_for_model
        url = get_base_url(info.ai_plugin.api.url)

        system_prefix = plugin.system_prefix + Template(open_ai_plugin_template).render(
            description_for_model=description_for_model,
            url=url,
            open_api=open_api,
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
        system_prefix: str,
        commands: dict[str, CommandConf],
    ) -> str:
        command_dict: CommandDict = {EndDialog.token(): EndDialog}

        for name, command_spec in commands.items():
            command_dict[name] = command_spec.implementation

        init_messages = [
            PLUGIN_SYSTEM_DIALOG_MESSAGE.format(
                commands=commands,
                system_prefix=system_prefix,
                query=self.query,
            )
        ]

        args = parse_args()
        model = create_chat_from_conf(args.openai_conf, args.chat_conf)

        chat = CommandChain(
            model_client=ModelClient(model=model),
            name="PLUGIN:" + self.name,
            resp_prompt=RESP_DIALOG_PROMPT,
            ctx=ExecutionContext(command_dict),
        )

        try:
            return chat.run_chat(init_messages, self.callback)
        except Exception as e:
            print_exception()
            return "ERROR: " + str(e)
