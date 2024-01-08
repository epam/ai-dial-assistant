from typing import Callable


class ArgsCallback:
    """Callback for reporting arguments"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    def on_args_start(self):
        self.callback("(")

    def on_args_chunk(self, chunk: str):
        self.callback(chunk)

    def on_args_end(self):
        self.callback(")")
