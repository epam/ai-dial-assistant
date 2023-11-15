from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from typing_extensions import override

from aidial_assistant.json_stream.characterstream import CharacterStream


class NodeResolver(ABC):
    @abstractmethod
    async def resolve(self, stream: CharacterStream) -> "JsonNode":
        pass


T = TypeVar("T")


class JsonNode(ABC, Generic[T]):
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
    def value(self) -> T:
        pass


class ComplexNode(JsonNode[T], ABC, Generic[T]):
    def __init__(self, char_position: int):
        super().__init__(char_position)

    @abstractmethod
    async def parse(
        self, stream: CharacterStream, dependency_resolver: NodeResolver
    ):
        pass


class PrimitiveNode(JsonNode[T], ABC, Generic[T]):
    @abstractmethod
    def raw_data(self) -> str:
        pass

    @override
    async def to_string_chunks(self) -> AsyncIterator[str]:
        yield self.raw_data()

    @staticmethod
    async def collect(stream: CharacterStream) -> str:
        raw_data = ""
        while True:
            char = await stream.apeek()
            if char.isspace() or char in ",:[]{}":
                return raw_data
            else:
                raw_data += char
                await stream.askip()
