from abc import ABC

from protocol.commands.base import ExecutionCallback


class ArgCallback:
    """Callback for reporting arguments"""
    def __init__(self, arg_index: int, callback: ExecutionCallback):
        self.arg_index = arg_index
        self.callback = callback

    async def on_arg_start(self):
        """Called when the arg starts"""
        await self.callback('"' if self.arg_index == 0 else ', "')

    async def on_arg(self, token: str):
        """Called when an argument token is read"""
        await self.callback(token.replace('"', '\\"'))

    async def on_arg_end(self):
        """Called when the arg ends"""
        await self.callback('"')
