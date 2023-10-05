from abc import abstractmethod
from typing import ContextManager

from aidial_assistant.chain.callbacks.args_callback import ArgsCallback
from aidial_assistant.commands.base import ExecutionCallback, ResultObject


class CommandCallback(ContextManager):
    """Callback for reporting a command"""

    @abstractmethod
    def on_command(self, command: str):
        """Called when a command is read"""

    @abstractmethod
    def args_callback(self) -> ArgsCallback:
        """Returns a callback for reporting arguments"""

    @abstractmethod
    def execution_callback(self) -> ExecutionCallback:
        """Returns a callback for reporting execution"""

    @abstractmethod
    def on_result(self, result: ResultObject):
        """Called when a result is read"""

    @abstractmethod
    def on_error(self, error: BaseException):
        """Called when an error occurs"""
