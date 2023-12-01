from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from typing_extensions import override

from aidial_assistant.json_stream.chunked_char_stream import ChunkedCharStream
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
)


class NodeResolver(ABC):
    @abstractmethod
    async def resolve(self, stream: ChunkedCharStream) -> "JsonNode":
        pass


TValue = TypeVar("TValue")
TElement = TypeVar("TElement")


class JsonNode(ABC, Generic[TValue]):
    def __init__(self, char_position: int):
        self._char_position = char_position

    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def to_string_chunks(self) -> AsyncIterator[str]:
        pass

    @property
    def char_position(self) -> int:
        return self._char_position

    @abstractmethod
    def value(self) -> TValue:
        pass


class CompoundNode(
    JsonNode[TValue], AsyncIterator[TElement], ABC, Generic[TValue, TElement]
):
    def __init__(self, source: AsyncIterator[TElement], char_position: int):
        super().__init__(char_position)
        self._source = source

    @override
    def __aiter__(self) -> AsyncIterator[TElement]:
        return self

    @override
    async def __anext__(self) -> TElement:
        result = await anext(self._source)
        self._accumulate(result)

        return result

    @abstractmethod
    def _accumulate(self, element: TElement):
        pass

    async def read_to_end(self):
        async for _ in self:
            pass


class AtomicNode(JsonNode[TValue], ABC, Generic[TValue]):
    def __init__(self, raw_data: str, char_position: int):
        super().__init__(char_position)
        self._raw_data = raw_data

    @override
    async def to_string_chunks(self) -> AsyncIterator[str]:
        yield self._raw_data

    @classmethod
    async def parse(cls, stream: ChunkedCharStream) -> "AtomicNode":
        position = stream.char_position
        return cls(await AtomicNode._read_all(stream), position)

    @staticmethod
    async def _read_all(stream: ChunkedCharStream) -> str:
        try:
            raw_data = ""
            while True:
                char = await stream.apeek()
                if char.isspace() or char in ",:[]{}":
                    return raw_data
                else:
                    raw_data += char
                    await stream.askip()
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)
