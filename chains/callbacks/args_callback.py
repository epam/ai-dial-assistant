from chains.callbacks.arg_callback import ArgCallback
from protocol.commands.base import ExecutionCallback


class ArgsCallback:
    """Callback for reporting arguments"""

    def __init__(self, callback: ExecutionCallback):
        self.callback = callback
        self.arg_index = -1

    async def on_args_start(self):
        """Called when the arguments start"""
        await self.callback("(")

    def arg_callback(self) -> ArgCallback:
        """Returns a callback for reporting an argument"""
        self.arg_index += 1
        return ArgCallback(self.arg_index, self.callback)

    async def on_args_end(self):
        """Called when the arguments end"""
        await self.callback(")\n")
