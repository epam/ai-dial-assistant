import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from enum import Enum
from typing import Any, List, TypedDict, Callable

from typing_extensions import override


class ExecutionCallback:
    """Callback for reporting execution"""

    def __init__(self, callback: Callable[[str], Awaitable[None]] = lambda token: None):
        self.callback = callback

    async def __call__(self, token: str):
        """Called when a command is executed"""
        await self.callback(token)


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

    async def execute(self, args: List[Any], execution_callback: ExecutionCallback) -> ResultObject:
        raise Exception(f"Command {self} isn't implemented")

    def __str__(self) -> str:
        return self.token()


class FinalCommand(Command, ABC):
    @override
    async def execute(self, args: List[Any], execution_callback: ExecutionCallback) -> ResultObject:
        raise Exception(f"Internal error: command {self} is final and can't be executed")


class CommandObject(TypedDict):
    command: str
    args: List[str]


def commands_to_text(commands: List[CommandObject]) -> str:
    return json.dumps({"commands": commands})


CommandConstructor = Callable[[], Command]


def resolve_constructor(implementation: str) -> CommandConstructor:
    parts = implementation.split("::")
    if len(parts) != 2:
        raise ValueError(
            f"Implementation is expected in the format of <module>::<class>, but got {implementation}"
        )

    module_name, class_name = parts

    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Failed to load module '{module_name}': {str(e)}")

    try:
        return getattr(plugin, class_name)
    except AttributeError as e:
        raise ValueError(
            f"Failed to load class '{class_name}' from '{module_name}': {str(e)}. Available classes: {dir(plugin)}"
        )
