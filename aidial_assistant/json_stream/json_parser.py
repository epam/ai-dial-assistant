from typing_extensions import override

from aidial_assistant.json_stream.chunked_char_stream import (
    ChunkedCharStream,
    skip_whitespaces,
)
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
    unexpected_symbol_error,
)
from aidial_assistant.json_stream.json_array import JsonArray
from aidial_assistant.json_stream.json_bool import JsonBoolean
from aidial_assistant.json_stream.json_node import JsonNode, NodeParser
from aidial_assistant.json_stream.json_null import JsonNull
from aidial_assistant.json_stream.json_number import JsonNumber
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_string import JsonString


def array_node(node: JsonNode) -> JsonArray:
    if not isinstance(node, JsonArray):
        raise TypeError(
            f"Expected json array at position {node.pos}, got {node.type}"
        )

    return node


def object_node(node: JsonNode) -> JsonObject:
    if not isinstance(node, JsonObject):
        raise TypeError(
            f"Expected json object at position {node.pos}, got {node.type}"
        )

    return node


def string_node(node: JsonNode) -> JsonString:
    if not isinstance(node, JsonString):
        raise TypeError(
            f"Expected json string at position {node.pos}, got {node.type}"
        )

    return node


class JsonParser(NodeParser):
    @override
    async def parse(self, stream: ChunkedCharStream) -> JsonNode:
        try:
            await skip_whitespaces(stream)
            char = await stream.apeek()
            if JsonObject.starts_with(char):
                return JsonObject.parse(stream, self)

            if JsonArray.starts_with(char):
                return JsonArray.parse(stream, self)

            if JsonString.starts_with(char):
                return JsonString.parse(stream)

            if JsonNumber.starts_with(char):
                return await JsonNumber.parse(stream)

            if JsonNull.starts_with(char):
                return await JsonNull.parse(stream)

            if JsonBoolean.starts_with(char):
                return await JsonBoolean.parse(stream)
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)

        raise unexpected_symbol_error(char, stream.char_position)
