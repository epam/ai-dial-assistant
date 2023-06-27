from asyncio import create_task

from typing_extensions import override

from chains.json_stream.parsing_context import ParsingContext
from chains.json_stream.json_array import JsonArray
from chains.json_stream.json_node import JsonNode, NodeResolver
from chains.json_stream.json_normalizer import JsonNormalizer
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_string import JsonString
from chains.json_stream.tokenator import Tokenator


def array_node(node: JsonNode) -> JsonArray:
    if not isinstance(node, JsonArray):
        raise Exception("Not an array node")

    return node


def object_node(node: JsonNode) -> JsonObject:
    if not isinstance(node, JsonObject):
        raise Exception("Not an object node")

    return node


def string_node(node: JsonNode) -> JsonString:
    if not isinstance(node, JsonString):
        raise Exception("Not a string node")

    return node


class RootNodeResolver(NodeResolver):
    @override
    async def resolve(self, stream: Tokenator) -> JsonNode:
        normalised_stream = JsonNormalizer(stream)
        char = await normalised_stream.apeek()
        if char == JsonObject.token():
            return JsonObject()

        if char == JsonString.token():
            return JsonString()

        if char == JsonArray.token():
            return JsonArray()

        raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")


class JsonParser:
    @staticmethod
    async def parse(stream: Tokenator) -> ParsingContext:
        node_resolver = RootNodeResolver()
        root = await node_resolver.resolve(stream)
        task = create_task(root.parse(stream, node_resolver))
        return ParsingContext(root, task)
