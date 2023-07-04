from asyncio import Queue
from typing import Any

from typing_extensions import override

from chains.callbacks.args_callback import ArgsCallback
from chains.callbacks.chain_callback import ChainCallback
from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback
from protocol.commands.base import ExecutionCallback
from utils.state import OpenAIRole, MessageField, CustomContentField, CommonField, StateField, StageField, StageStatus


def custom_content(content: dict[str, Any]):
    return {MessageField.CUSTOM_CONTENT: content}


def stage(index: int, content: dict[str, Any]):
    return custom_content({CustomContentField.STAGES: [{CommonField.INDEX: index} | content]})


def state(index: int, content: dict[str, Any]):
    return custom_content({CustomContentField.STATE: {StateField.INVOCATIONS: [{CommonField.INDEX: index} | content]}})


class ServerCommandCallback(CommandCallback):
    def __init__(self, command_index: int, queue: Queue[Any]):
        self.command_index = command_index
        self.queue = queue

    @override
    async def on_command(self, command: str):
        await self._on_stage_name(command)

    @override
    def execution_callback(self) -> ExecutionCallback:
        return ExecutionCallback(self._on_stage_content)

    @override
    def args_callback(self) -> ArgsCallback:
        return ArgsCallback(ExecutionCallback(self._on_stage_name))

    @override
    async def on_result(self, response):
        # Result reported by plugin
        await self._on_stage({StageField.STATUS: StageStatus.COMPLETED})

    async def on_error(self, error: Exception):
        await self._on_stage({CommonField.CONTENT: f"\n{str(error)}", StageField.STATUS: StageStatus.FAILED})

    async def _on_stage(self, delta: dict[str, Any]):
        await self.queue.put(stage(self.command_index, delta))

    async def _on_stage_content(self, token: str):
        await self._on_stage({CommonField.CONTENT: token})

    async def _on_stage_name(self, token: str):
        await self._on_stage({StageField.NAME: token})


class ServerResultCallback(ResultCallback):
    def __init__(self, queue: Queue[Any]):
        self.queue = queue

    async def on_result(self, token):
        await self.queue.put({CommonField.CONTENT: token})


class ServerChainCallback(ChainCallback):
    def __init__(self):
        self.stage_index: int = -1
        self.invocation_index: int = -1
        self.queue = Queue[Any | None]()

    @override
    async def on_start(self):
        await self.queue.put({MessageField.ROLE: OpenAIRole.ASSISTANT})

    @override
    def command_callback(self) -> CommandCallback:
        self.stage_index += 1
        return ServerCommandCallback(self.stage_index, self.queue)

    @override
    async def on_state(self, request: str, response: str):
        self.invocation_index += 1
        await self.queue.put(
            state(self.invocation_index, {StateField.REQUEST: request, StateField.RESPONSE: response}))

    @override
    def result_callback(self) -> ResultCallback:
        return ServerResultCallback(self.queue)

    @override
    async def on_end(self):
        await self.queue.put(None)

    @override
    async def on_error(self, error: Exception):
        self.stage_index += 1
        await self.queue.put(
            stage(
                self.stage_index,
                {CommonField.CONTENT: f"Error: {str(error)}\n", StageField.STATUS: StageStatus.FAILED}))
