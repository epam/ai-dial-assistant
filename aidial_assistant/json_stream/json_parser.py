from typing_extensions import override

from aidial_assistant.json_stream.characterstream import CharacterStream
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
    unexpected_symbol_error,
)
from aidial_assistant.json_stream.json_array import JsonArray
from aidial_assistant.json_stream.json_bool import JsonBoolean
from aidial_assistant.json_stream.json_node import (
    JsonNode,
    NodeResolver,
    PrimitiveNode,
)
from aidial_assistant.json_stream.json_normalizer import JsonNormalizer
from aidial_assistant.json_stream.json_null import JsonNull
from aidial_assistant.json_stream.json_number import JsonNumber
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_string import JsonString


def array_node(node: JsonNode) -> JsonArray:
    if not isinstance(node, JsonArray):
        raise TypeError(
            f"Expected json array at position {node.char_position}, got {node.type}"
        )

    return node


def object_node(node: JsonNode) -> JsonObject:
    if not isinstance(node, JsonObject):
        raise TypeError(
            f"Expected json object at position {node.char_position}, got {node.type}"
        )

    return node


def string_node(node: JsonNode) -> JsonString:
    if not isinstance(node, JsonString):
        raise TypeError(
            f"Expected json string at position {node.char_position}, got {node.type}"
        )

    return node


class RootNodeResolver(NodeResolver):
    @override
    async def resolve(self, stream: CharacterStream) -> JsonNode:
        try:
            normalised_stream = JsonNormalizer(stream)
            char = await normalised_stream.apeek()
            if char == JsonObject.token():
                return JsonObject.parse(stream, self)

            if char == JsonArray.token():
                return JsonArray.parse(stream, self)

            if char == JsonString.token():
                return JsonString.parse(stream)

            if JsonNumber.is_number(char):
                position = stream.char_position
                return JsonNumber(await PrimitiveNode.collect(stream), position)

            if JsonNull.is_null(char):
                position = stream.char_position
                return JsonNull(await PrimitiveNode.collect(stream), position)

            if JsonBoolean.is_bool(char):
                position = stream.char_position
                return JsonBoolean(
                    await PrimitiveNode.collect(stream), position
                )
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)

        raise unexpected_symbol_error(char, stream.char_position)


class JsonParser:
    @staticmethod
    async def parse(stream: CharacterStream) -> JsonNode:
        node_resolver = RootNodeResolver()

        return await node_resolver.resolve(stream)
