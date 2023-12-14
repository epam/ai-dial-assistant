from abc import ABC
from typing import Any

from aidial_assistant.commands.base import ExecutionCallback


class ToolRunner(ABC):
    async def run(
        self, name: str, arg: Any, execution_callback: ExecutionCallback
    ):
        pass
