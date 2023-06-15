import importlib
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class Command(ABC):
    dict: Dict

    @staticmethod
    @abstractmethod
    def token() -> str:
        pass

    def json(self) -> Dict:
        return self.dict

    def execute(self) -> Any:
        raise Exception(f"{self.print()} isn't implemented")

    def __eq__(self, other):
        return self.print() == other.print()

    def print(self) -> str:
        """Print the full object"""
        return json.dumps(self.dict, indent=2)

    def print_header(self) -> str:
        """Print short version of the command: no thought, no payload"""
        return f"{self.token()}({','.join(self.get_args())})"

    def get_args(self) -> list[str]:
        return self.dict.get("args", [])

    def __str__(self) -> str:
        return self.print_header()


CommandConstructor = Callable[[dict], Command]


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
