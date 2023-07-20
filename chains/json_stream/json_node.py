from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar, Generic

from typing_extensions import override

from chains.json_stream.tokenator import Tokenator


def unexpected_symbol_error(char: str, char_position: int) -> ValueError:
    return ValueError(f"Failed to parse json string: unexpected symbol {char} at position {char_position}")


class NodeResolver(ABC):
    @abstractmethod
    async def resolve(self, stream: Tokenator) -> 'JsonNode':
        pass


T = TypeVar('T')


class JsonNode(ABC, Generic[T]):
    def __init__(self, char_position: int):
        self._char_position = char_position

    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    async def to_string_tokens(self) -> AsyncIterator[str]:
        pass

    @property
    def char_position(self) -> int:
        return self._char_position

    @abstractmethod
    def value(self) -> T:
        pass


class ComplexNode(JsonNode[T], ABC, Generic[T]):
    def __init__(self, char_position: int):
        super().__init__(char_position)

    @abstractmethod
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        pass

    @staticmethod
    def throw_if_exception(entry):
        if isinstance(entry, StopAsyncIteration):
            raise Exception("Failed to parse json: unexpected end of stream.")

        if isinstance(entry, BaseException):
            raise entry

        return entry


class PrimitiveNode(JsonNode[T], ABC, Generic[T]):
    @abstractmethod
    def raw_data(self) -> str:
        pass

    @override
    async def to_string_tokens(self) -> AsyncIterator[str]:
        yield self.raw_data()

    @staticmethod
    async def collect(stream: Tokenator) -> str:
        raw_data = ''
        while True:
            char = await stream.apeek()
            if char.isspace() or char in ',:[]{}':
                return raw_data
            else:
                raw_data += char
                await stream.askip()
