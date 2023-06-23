import importlib
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, List, TypedDict

from typing_extensions import override


class ExecutionCallback(ABC):
    @abstractmethod
    def on_message(self, token: str):
        pass


class Command(ABC):
    @staticmethod
    @abstractmethod
    def token() -> str:
        pass

    def execute(self, args: List[Any], execution_callback: ExecutionCallback) -> Any:
        raise Exception(f"Command {self} isn't implemented")

    def __str__(self) -> str:
        return self.token()


class FinalCommand(Command, ABC):
    @override
    def execute(self, args: List[Any], execution_callback: ExecutionCallback) -> Any:
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
