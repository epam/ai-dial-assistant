from types import TracebackType
from typing import Any, Callable, Awaitable

from aidial_sdk.chat_completion.choice import Choice
from aidial_sdk.chat_completion.stage import Stage
from typing_extensions import override

from chains.callbacks.arg_callback import ArgCallback
from chains.callbacks.args_callback import ArgsCallback
from chains.callbacks.chain_callback import ChainCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback
from protocol.commands.base import ExecutionCallback, ResultObject
from protocol.commands.run_plugin import RunPlugin
from utils.state import (
    CommonField,
    StateField,
)


def state(index: int, content: dict[str, Any]):
    return {StateField.INVOCATIONS: [{CommonField.INDEX: index} | content]}


class PluginNameArgCallback(ArgCallback):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__(0, callback)

    @override
    def on_arg(self, token: str):
        token = token.replace('"', "")
        if len(token) > 0:
            self.callback(token)

    @override
    def on_arg_end(self):
        self.callback("(")


class RunPluginArgsCallback(ArgsCallback):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__(callback)

    @override
    async def on_args_start(self):
        pass

    @override
    def arg_callback(self) -> ArgCallback:
        self.arg_index += 1
        if self.arg_index == 0:
            return PluginNameArgCallback(self.callback)
        else:
            return ArgCallback(self.arg_index - 1, self.callback)


class ServerCommandCallback(CommandCallback):
    def __init__(self, stage: Stage):
        self.stage = stage
        self._args_callback = ArgsCallback(self._on_stage_name)

    @override
    def on_command(self, command: str):
        if command == RunPlugin.token():
            self._args_callback = RunPluginArgsCallback(self._on_stage_name)
        else:
            self._on_stage_name(command)

    @override
    def execution_callback(self) -> ExecutionCallback:
        return ExecutionCallback(self._on_stage_content)

    @override
    def args_callback(self) -> ArgsCallback:
        return self._args_callback

    @override
    def on_result(self, result: ResultObject):
        # Result reported by plugin
        pass

    @override
    def on_error(self, error: BaseException):
        self.stage.append_content(f"\n{str(error)}")

    def _on_stage_name(self, token: str):
        self.stage.append_name(token)

    def _on_stage_content(self, token: str):
        self.stage.append_content(token)

    def __enter__(self):
        self.stage.__enter__()
        return self

    @override
    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ):
        if __exc_value is not None:
            self.on_error(__exc_value)

        self.stage.__exit__(__exc_type, __exc_value, __traceback)


class ServerResultCallback(ResultCallback):
    def __init__(self, choice: Choice):
        self.choice = choice

    def on_result(self, token):
        self.choice.append_content(token)


class ServerChainCallback(ChainCallback):
    def __init__(self, choice: Choice):
        self.invocation_index: int = -1
        self.choice = choice

    @override
    def command_callback(self) -> CommandCallback:
        return ServerCommandCallback(self.choice.create_stage())

    @override
    def on_state(self, request: str, response: str):
        self.invocation_index += 1
        self.choice.append_state(
            state(
                self.invocation_index,
                {StateField.REQUEST: request, StateField.RESPONSE: response},
            )
        )

    @override
    def result_callback(self) -> ResultCallback:
        return ServerResultCallback(self.choice)

    @override
    def on_error(self, title: str, error: Exception):
        with self.choice.create_stage(title) as stage:
            stage.append_content(f"Error: {str(error)}\n")
