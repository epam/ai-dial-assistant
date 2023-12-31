from typing import Callable


class ArgCallback:
    """Callback for reporting arguments"""

    def __init__(self, arg_index: int, callback: Callable[[str], None]):
        self.arg_index = arg_index
        self.callback = callback

    def on_arg_start(self):
        """Called when the arg starts"""
        if self.arg_index > 0:
            self.callback(", ")

    def on_arg(self, chunk: str):
        """Called when an argument chunk is read"""
        self.callback(chunk)

    def on_arg_end(self):
        """Called when the arg ends"""
