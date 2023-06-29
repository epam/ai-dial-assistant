from abc import ABC, abstractmethod

from chains.json_stream.tokenator import Tokenator


class NodeResolver(ABC):
    @abstractmethod
    async def resolve(self, stream: Tokenator) -> 'JsonNode':
        pass


class JsonNode(ABC):
    def __init__(self, char_position: int):
        self._char_position = char_position

    @abstractmethod
    def type(self) -> str:
        pass

    @property
    def char_position(self) -> int:
        return self._char_position

    @abstractmethod
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        pass

    @staticmethod
    def throw_if_exception(entry):
        if isinstance(entry, BaseException):
            raise entry

        return entry

