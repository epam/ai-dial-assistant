from abc import abstractmethod

from chains.callbacks.command_callback import CommandCallback
from chains.callbacks.result_callback import ResultCallback


class ChainCallback:
    """Callback for reporting command chain"""

    async def on_start(self):
        """Called when the chain starts"""

    def command_callback(self) -> CommandCallback:
        """Returns a callback for reporting a command"""
        return CommandCallback()

    async def on_end(self):
        """Called when the chain ends"""

    async def on_state(self, request: str, response: str):
        """Report an AI message"""

    def result_callback(self) -> ResultCallback:
        """Returns a callback for reporting a result"""
        return ResultCallback()

    async def on_error(self, title: str, error: Exception):
        """Called when an error occurs"""
