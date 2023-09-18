from abc import abstractmethod
from typing import ContextManager

from chains.callbacks.args_callback import ArgsCallback
from protocol.commands.base import ExecutionCallback


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
    def on_result(self, response):
        """Called when a result is read"""

    @abstractmethod
    def on_error(self, error: BaseException):
        """Called when an error occurs"""
