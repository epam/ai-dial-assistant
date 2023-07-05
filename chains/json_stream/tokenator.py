from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from typing_extensions import override


T = TypeVar("T")


class AsyncPeekable(ABC, Generic[T], AsyncIterator[T]):
    @abstractmethod
    async def apeek(self) -> T:
        pass

    async def askip(self) -> None:
        await anext(self)


class Tokenator(AsyncPeekable[str]):
    def __init__(self, source: AsyncIterator[str]):
        self._source = source
        self._token: str = ""
        self._next_char_offset: int = 0
        self._token_position: int = 0

    @override
    def __aiter__(self) -> AsyncIterator[str]:
        return self

    @override
    async def __anext__(self) -> str:
        result = await self.apeek()
        self._next_char_offset += 1
        return result

    @override
    async def apeek(self) -> str:
        while self._next_char_offset == len(self._token):
            self._token_position += len(self._token)
            self._token = await anext(self._source)  # type: ignore
            self._next_char_offset = 0
        return self._token[self._next_char_offset]

    @property
    def token_position(self) -> int:
        return self._token_position

    @property
    def char_position(self) -> int:
        return self._token_position + self._next_char_offset
