from types import TracebackType
from typing import Callable

from typing_extensions import override

from aidial_assistant.chain.callbacks.args_callback import ArgsCallback
from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.commands.base import (
    ExecutionCallback,
    ResultObject,
    ResultType,
)


class PluginCommandCallback(CommandCallback):
    def __init__(self, callback: ExecutionCallback):
        self.callback = callback

    @override
    def on_command(self, command: str):
        self.callback(f"```javascript\n{command}")

    @override
    def args_callback(self) -> ArgsCallback:
        return ArgsCallback(self.callback)

    @override
    def execution_callback(self) -> ExecutionCallback:
        return self.callback

    @override
    def on_result(self, result: ResultObject):
        syntax = "json" if result.type == ResultType.JSON else "text"
        self.callback(f"\n```\n```{syntax}\n{result.text}\n```\n")

    @override
    def on_error(self, error: BaseException):
        self.callback(f"\n```\n```\nError: {str(error)}\n```\n")

    @override
    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ):
        if __exc_value is not None:
            self.on_error(__exc_value)


class PluginResultCallback(ResultCallback):
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    @override
    def on_result(self, token):
        self.callback(token)


class PluginChainCallback(ChainCallback):
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    @override
    def command_callback(self) -> PluginCommandCallback:
        return PluginCommandCallback(self.callback)

    @override
    def result_callback(self) -> ResultCallback:
        return PluginResultCallback(self.callback)

    @override
    def on_state(self, request: str, response: str):
        # Plugin state is not currently supported
        pass

    @override
    def on_error(self, title: str, error: Exception):
        pass
