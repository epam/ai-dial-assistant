from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, TypedDict

from typing_extensions import override


class ResultType(str, Enum):
    TEXT = "Text"
    JSON = "JSon"


class ResultObject:
    def __init__(self, result_type: ResultType, text: str):
        self._type = result_type
        self._text = text

    @property
    def type(self) -> ResultType:
        return self._type

    @property
    def text(self) -> str:
        return self._text


class ExecutionCallback:
    """Callback for reporting execution"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    def on_token(self, token: str):
        self.callback(token)


class TextResult(ResultObject):
    def __init__(self, text: str):
        super().__init__(ResultType.TEXT, text)


class JsonResult(ResultObject):
    def __init__(self, text: str):
        super().__init__(ResultType.JSON, text)


class Command(ABC):
    @staticmethod
    @abstractmethod
    def token() -> str:
        pass

    async def execute(
        self, args: List[Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        raise Exception(f"Command {self} isn't implemented")

    def __str__(self) -> str:
        return self.token()

    def assert_arg_count(self, args: List[Any], count: int):
        if len(args) != count:
            raise ValueError(
                f"Command {self} expects {count} args, but got {len(args)}"
            )


class FinalCommand(Command, ABC):
    @override
    async def execute(
        self, args: List[Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        raise Exception(
            f"Internal error: command {self} is final and can't be executed"
        )


class CommandObject(TypedDict):
    command: str
    args: List[str]


CommandConstructor = Callable[[], Command]
