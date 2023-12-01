from abc import ABC
from collections.abc import AsyncIterator

from typing_extensions import override


class ChunkedCharStream(ABC, AsyncIterator[str]):
    def __init__(self, source: AsyncIterator[str]):
        self._source = source
        self._chunk: str = ""
        self._next_char_offset: int = 0
        self._chunk_position: int = 0

    @override
    def __aiter__(self) -> AsyncIterator[str]:
        return self

    @override
    async def __anext__(self) -> str:
        result = await self.apeek()
        self._next_char_offset += 1
        return result

    async def apeek(self) -> str:
        while self._next_char_offset == len(self._chunk):
            self._chunk_position += len(self._chunk)
            self._chunk = await anext(self._source)  # type: ignore
            self._next_char_offset = 0
        return self._chunk[self._next_char_offset]

    async def askip(self):
        await anext(self)

    async def skip_whitespaces(self) -> "ChunkedCharStream":
        while True:
            char = await self.apeek()
            if not str.isspace(char):
                break
            await self.askip()

        return self

    @property
    def chunk_position(self) -> int:
        return self._chunk_position

    @property
    def char_position(self) -> int:
        return self._chunk_position + self._next_char_offset
