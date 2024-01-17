import json
from typing import Any, AsyncIterator
from unittest.mock import MagicMock, Mock

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from typing_extensions import override

from aidial_assistant.chain.callbacks.chain_callback import ChainCallback
from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback
from aidial_assistant.chain.command_chain import (
    LimitExceededException,
    ModelRequestLimiter,
)
from aidial_assistant.commands.base import (
    Command,
    ExecutionCallback,
    ResultObject,
    ResultType,
)
from aidial_assistant.model.model_client import (
    ExtraResultsCallback,
    ModelClient,
)


class TestModelRequestLimiter(ModelRequestLimiter):
    def __init__(self, exception_trigger: list[ChatCompletionMessageParam]):
        self.exception_trigger = exception_trigger

    async def verify_limit(self, messages: list[ChatCompletionMessageParam]):
        if messages == self.exception_trigger:
            raise LimitExceededException()


class TestModelClient(ModelClient):
    def __init__(
        self,
        tool_calls: dict[str, list[ChatCompletionMessageToolCallParam]],
        results: dict[str, str],
    ):
        super().__init__(Mock(), {})
        self.tool_calls = tool_calls
        self.results = results

    @override
    async def agenerate(
        self,
        messages: list[ChatCompletionMessageParam],
        extra_results_callback: ExtraResultsCallback | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        args = TestModelClient.agenerate_key(messages, **kwargs)
        if extra_results_callback and args in self.tool_calls:
            extra_results_callback.on_tool_calls(self.tool_calls[args])
            return

        if args in self.results:
            yield self.results[args]
            return

        assert False, f"Unexpected arguments: {args}"

    @staticmethod
    def agenerate_key(
        messages: list[ChatCompletionMessageParam], **kwargs
    ) -> str:
        return json.dumps({"messages": messages, **kwargs})


class TestCommand(Command):
    def __init__(self, results: dict[str, str]):
        self.results = results

    @staticmethod
    def token() -> str:
        return "test-command"

    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        args_string = TestCommand.execute_key(args)
        assert args_string in self.results, f"Unexpected argument: {args}"

        return ResultObject(ResultType.TEXT, self.results[args_string])

    @staticmethod
    def execute_key(args: dict[str, Any]) -> str:
        return json.dumps({"args": args})


class TestResultCallback(ResultCallback):
    def __init__(self):
        self.result: str = ""

    def on_result(self, chunk: str):
        self.result += chunk


class TestChainCallback(ChainCallback):
    def __init__(self):
        self.mock_result_callback = TestResultCallback()

    def command_callback(self) -> CommandCallback:
        return MagicMock(spec=CommandCallback)

    def on_state(self, request: str, response: str):
        pass

    def result_callback(self) -> ResultCallback:
        return self.mock_result_callback

    def on_error(self, title: str, error: str):
        pass
