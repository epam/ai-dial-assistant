from abc import ABC, abstractmethod

from aidial_assistant.chain.callbacks.command_callback import CommandCallback
from aidial_assistant.chain.callbacks.result_callback import ResultCallback


class ChainCallback(ABC):
    """Callback for reporting command chain"""

    @abstractmethod
    def command_callback(self) -> CommandCallback:
        """Returns a callback for reporting a command"""

    @abstractmethod
    def on_state(self, request: str, response: str):
        """Report an AI message"""

    @abstractmethod
    def result_callback(self) -> ResultCallback:
        """Returns a callback for reporting a result"""

    @abstractmethod
    def on_error(self, title: str, error: Exception):
        """Called when an error occurs"""
