from typing import Callable

from aidial_assistant.chain.callbacks.arg_callback import ArgCallback


class ArgsCallback:
    """Callback for reporting arguments"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.arg_index = -1

    def on_args_start(self):
        """Called when the arguments start"""
        self.callback("(")

    def arg_callback(self) -> ArgCallback:
        """Returns a callback for reporting an argument"""
        self.arg_index += 1
        return ArgCallback(self.arg_index, self.callback)

    def on_args_end(self):
        """Called when the arguments end"""
        self.callback(")")
