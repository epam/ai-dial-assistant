import json
from typing import Callable, Any


class ArgsCallback:
    """Callback for reporting arguments"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    def on_args(self, args: dict[str, Any]):
        """Called when the argument dict is constructed"""
        self.callback("(" + json.dumps(args) + ")")
