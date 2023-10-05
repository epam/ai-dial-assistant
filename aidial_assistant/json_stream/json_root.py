import asyncio
from typing import Any, AsyncIterator

from typing_extensions import override

from aidial_assistant.json_stream.json_array import JsonArray
from aidial_assistant.json_stream.json_bool import JsonBoolean
from aidial_assistant.json_stream.json_node import (
    ComplexNode,
    JsonNode,
    NodeResolver,
    PrimitiveNode,
    unexpected_symbol_error,
)
from aidial_assistant.json_stream.json_normalizer import JsonNormalizer
from aidial_assistant.json_stream.json_null import JsonNull
from aidial_assistant.json_stream.json_number import JsonNumber
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.json_stream.tokenator import Tokenator


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

        if JsonNumber.is_number(char):
            position = stream.char_position
            return JsonNumber(await PrimitiveNode.collect(stream), position)

        if JsonNull.is_null(char):
            position = stream.char_position
            return JsonNull(await PrimitiveNode.collect(stream), position)

        if JsonBoolean.is_bool(char):
            position = stream.char_position
            return JsonBoolean(await PrimitiveNode.collect(stream), position)

        raise unexpected_symbol_error(char, stream.char_position)


class JsonRoot(ComplexNode[Any]):
    def __init__(self):
        super().__init__(0)
        self._node: JsonNode | BaseException | None = None
        self._event = asyncio.Event()

    async def node(self) -> JsonNode:
        await self._event.wait()
        if self._node is None:
            # Should never happen
            raise Exception("Node was not parsed")

        return ComplexNode.throw_if_exception(self._node)

    @override
    def type(self) -> str:
        return "root"

    @override
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        try:
            self._node = await dependency_resolver.resolve(stream)
        except BaseException as e:
            self._node = e
        finally:
            self._event.set()

    @override
    async def to_string_tokens(self) -> AsyncIterator[str]:
        node = await self.node()
        async for token in node.to_string_tokens():  # type: ignore
            yield token

    @override
    def value(self) -> Any:
        if isinstance(self._node, JsonNode):
            return self._node.value()

        return None
