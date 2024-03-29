from types import TracebackType

from aidial_sdk.chat_completion import Status
from aidial_sdk.chat_completion.choice import Choice
from aidial_sdk.chat_completion.stage import Stage
from typing_extensions import override

from aidial_assistant.chain.callbacks.args_callback import ArgsCallback
from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.commands.base import ExecutionCallback, ResultObject
from aidial_assistant.utils.state import Invocation


class AssistantCommandCallback(CommandCallback):
    def __init__(self, stage: Stage, addon_name_mapping: dict[str, str]):
        self.stage = stage
        self.addon_name_mapping = addon_name_mapping

        self._args_callback = ArgsCallback(self._on_stage_name)

    @override
    def on_command(self, command: str):
        self._on_stage_name(self.addon_name_mapping.get(command, command))

    @override
    def execution_callback(self) -> ExecutionCallback:
        return self._on_stage_content

    @override
    def args_callback(self) -> ArgsCallback:
        return ArgsCallback(self._on_stage_name)

    @override
    def on_result(self, result: ResultObject):
        # Result reported by plugin
        pass

    @override
    def on_error(self, error: BaseException):
        self.stage.append_content(f"\n{str(error)}")

    def _on_stage_name(self, chunk: str):
        self.stage.append_name(chunk)

    def _on_stage_content(self, chunk: str):
        self.stage.append_content(chunk)

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


class AssistantResultCallback(ResultCallback):
    def __init__(self, choice: Choice):
        self.choice = choice

    def on_result(self, chunk: str):
        self.choice.append_content(chunk)


class AssistantChainCallback(ChainCallback):
    def __init__(self, choice: Choice, addon_name_mapping: dict[str, str]):
        self.choice = choice
        self.addon_name_mapping = addon_name_mapping

        self._invocations: list[Invocation] = []
        self._invocation_index: int = -1
        self._discarded_messages: int = 0

    @override
    def command_callback(self) -> CommandCallback:
        return AssistantCommandCallback(
            self.choice.create_stage(), self.addon_name_mapping
        )

    @override
    def on_state(self, request: str, response: str):
        self._invocation_index += 1
        self._invocations.append(
            Invocation(
                index=self._invocation_index, request=request, response=response
            )
        )

    @override
    def result_callback(self) -> ResultCallback:
        return AssistantResultCallback(self.choice)

    @override
    def on_error(self, title: str, error: str):
        stage = self.choice.create_stage(title)
        stage.open()
        stage.append_content(f"Error: {error}\n")
        stage.close(Status.FAILED)

    @property
    def invocations(self) -> list[Invocation]:
        return self._invocations
