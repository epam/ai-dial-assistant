from chains.callbacks.args_callback import ArgsCallback
from protocol.commands.base import ExecutionCallback


class CommandCallback:
    """Callback for reporting a command"""

    async def on_command(self, command: str):
        """Called when a command is read"""

    def args_callback(self) -> ArgsCallback:
        """Returns a callback for reporting arguments"""
        return ArgsCallback(ExecutionCallback())

    def execution_callback(self) -> ExecutionCallback:
        """Returns a callback for reporting execution"""
        return ExecutionCallback()

    async def on_result(self, response):
        """Called when a result is read"""

    async def on_error(self, error: Exception):
        """Called when an error occurs"""
