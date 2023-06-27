from abc import abstractmethod


class ResultCallback:
    """Callback for reporting a result"""

    async def on_start(self):
        """Called when the result starts"""

    async def on_result(self, token):
        """Called when a result token is read"""

    async def on_end(self):
        """Called when the result ends"""
