from typing import AsyncIterator


class CumulativeStream(AsyncIterator[str]):
    def __init__(self, stream: AsyncIterator[str]):
        self.stream = stream
        self.buffer = ""

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        chunk = await anext(self.stream)
        self.buffer += chunk
        return chunk
