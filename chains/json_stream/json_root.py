import asyncio

from typing_extensions import override

from chains.json_stream.json_array import JsonArray
from chains.json_stream.json_node import JsonNode, NodeResolver
from chains.json_stream.json_normalizer import JsonNormalizer
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_string import JsonString
from chains.json_stream.tokenator import Tokenator

class RootNodeResolver(NodeResolver):
    @override
    async def resolve(self, stream: Tokenator) -> JsonNode:
        normalised_stream = JsonNormalizer(stream)
        char = await normalised_stream.apeek()
        if char == JsonObject.token():
            return JsonObject(stream.char_position)

        if char == JsonString.token():
            return JsonString(stream.char_position)

        if char == JsonArray.token():
            return JsonArray(stream.char_position)

        raise ValueError(f"Unexpected symbol: {char} at {stream.char_position}")


class JsonRoot(JsonNode):
    def __init__(self):
        super().__init__(0)
        self._node: JsonNode | Exception | None = None
        self._event = asyncio.Event()

    async def node(self) -> JsonNode:
        await self._event.wait()
        if self._node is None:
            raise Exception("Node is not parsed")

        return JsonNode.throw_if_exception(self._node)

    def type(self) -> str:
        return "root"

    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        self._node = await dependency_resolver.resolve(stream)
        self._event.set()


