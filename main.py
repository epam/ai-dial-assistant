#!/usr/bin/env python3
from utils.init import init

init()

import asyncio

from chains.model_client import ModelClient
import os
import sys
from pathlib import Path

from chains.command_chain import CommandChain
from chains.callbacks.chain_callback import ChainCallback
from cli.main_args import parse_args
from conf.project_conf import (
    CommandConf,
    Conf,
    PluginCommand,
    PluginConf,
    PluginOpenAI,
    PluginTool,
    read_conf,
)
from llm.base import create_chat_from_conf
from prompts.dialog import RESP_DIALOG_PROMPT, SYSTEM_DIALOG_MESSAGE
from protocol.commands.run_plugin import RunPlugin
from protocol.commands.say_or_ask import SayOrAsk
from protocol.execution_context import CommandDict, ExecutionContext
from utils.open_ai_plugin import get_open_ai_plugin_info
from utils.optional import or_else


def collect_plugin(
    available_commands: dict[str, CommandConf],
    name: str,
    plugin: PluginConf,
    commands: dict[str, CommandConf],
    tools: dict[str, PluginTool],
    command_dict: CommandDict,
):
    if isinstance(plugin, PluginCommand):
        if name not in available_commands:
            raise ValueError(
                f"Unknown command: {name}. Available commands: {available_commands.keys()}"
            )
        command = available_commands[name]
        commands[name] = command
        command_dict[name] = command.implementation

    elif isinstance(plugin, PluginTool):
        tools[name] = plugin
    elif isinstance(plugin, PluginOpenAI):
        # Displaying OpenAI plugin as a regular tool in the main session
        info = get_open_ai_plugin_info(plugin.url)
        tools[name] = PluginTool(
            type="tool",
            system_prefix=plugin.system_prefix,
            description=or_else(
                info.open_api.info.description, info.ai_plugin.description_for_human
            ),
            commands=[],
        )


async def main() -> None:
    # Need to add this so that plugin modules residing at "plugins" folder could be references in index.yaml using relative module paths. Otherwise, we need to prefix all module paths with "plugins."
    sys.path.append(os.path.abspath("plugins"))

    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)

    conf = read_conf(Conf, Path("plugins/index.yaml"))

    commands: dict[str, CommandConf] = {}
    tools: dict[str, PluginTool] = {}

    command_dict: CommandDict = {
        RunPlugin.token(): lambda: RunPlugin(tools),
        SayOrAsk.token(): SayOrAsk,
    }

    for name, plugin in conf.plugins.items():
        collect_plugin(conf.commands, name, plugin, commands, tools, command_dict)

    init_messages = [
        SYSTEM_DIALOG_MESSAGE.format(
            system_prefix=conf.system_prefix, commands=commands, tools=tools
        )
    ]

    chain = CommandChain(
        name="MAIN",
        model_client=ModelClient(model=model),
        resp_prompt=RESP_DIALOG_PROMPT,
        ctx=ExecutionContext(command_dict),
    )

    await chain.run_chat(init_messages, ChainCallback())


if __name__ == "__main__":
    asyncio.run(main())
