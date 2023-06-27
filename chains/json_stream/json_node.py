from abc import ABC, abstractmethod

from chains.json_stream.tokenator import Tokenator


class NodeResolver(ABC):
    @abstractmethod
    async def resolve(self, stream: Tokenator) -> 'JsonNode':
        pass


class JsonNode(ABC):
    @abstractmethod
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        pass

    @staticmethod
    def throw_if_exception(entry):
        if isinstance(entry, Exception):
            raise entry

        return entry

