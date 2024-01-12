from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, TypedDict, TypeVar

from typing_extensions import override

ExecutionCallback = Callable[[str], None]


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
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        raise Exception(f"Command {self} isn't implemented")

    def __str__(self) -> str:
        return self.token()


class FinalCommand(Command, ABC):
    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        raise Exception(
            f"Internal error: command {self} is final and can't be executed"
        )


class CommandObject(TypedDict):
    command: str
    args: List[str]


CommandConstructor = Callable[[], Command]


T = TypeVar("T")


def get_required_field(args: dict[str, T], field: str) -> T:
    value = args.get(field)
    if value is None:
        raise Exception(f"Parameter '{field}' is required")
    return value
