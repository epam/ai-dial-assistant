import json

from typing_extensions import override

from chains.callbacks.args_callback import ArgsCallback
from chains.callbacks.chain_callback import ChainCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback
from protocol.commands.base import ExecutionCallback, ResultObject, ResultType


class PluginCommandCallback(CommandCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    @override
    async def on_command(self, command: str):
        await self.callback(f"```javascript\n{command}")

    @override
    def args_callback(self) -> ArgsCallback:
        return ArgsCallback(self.callback)

    @override
    def execution_callback(self) -> ExecutionCallback:
        return self.callback

    @override
    async def on_result(self, result: ResultObject):
        syntax = "json" if result.type == ResultType.JSON else "text"
        await self.callback(f"\n```\n```{syntax}\n{result.text}\n```\n")

    @override
    async def on_error(self, error: Exception):
        await self.callback(f"\n```\n```\n**Error:** {str(error)}\n```\n")


class PluginResultCallback(ResultCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    @override
    async def on_result(self, token):
        await self.callback(token)


class PluginChainCallback(ChainCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    @override
    def command_callback(self) -> PluginCommandCallback:
        return PluginCommandCallback(self.callback)

    @override
    def result_callback(self) -> ResultCallback:
        return PluginResultCallback(self.callback)
