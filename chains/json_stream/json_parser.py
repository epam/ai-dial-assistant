import json
from asyncio import create_task
from collections.abc import AsyncIterator

from chains.json_stream.json_array import JsonArray
from chains.json_stream.json_node import JsonNode, PrimitiveNode, ComplexNode
from chains.json_stream.json_number import JsonNumber
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_root import JsonRoot, RootNodeResolver
from chains.json_stream.json_string import JsonString
from chains.json_stream.parsing_context import ParsingContext
from chains.json_stream.tokenator import Tokenator


def array_node(node: JsonNode) -> JsonArray:
    if not isinstance(node, JsonArray):
        raise ValueError(f"Expected json array at position {node.char_position}, got {node.type}")

    return node


def object_node(node: JsonNode) -> JsonObject:
    if not isinstance(node, JsonObject):
        raise ValueError(f"Expected json object at position {node.char_position}, got {node.type}")

    return node


def string_node(node: JsonNode) -> JsonString:
    if not isinstance(node, JsonString):
        raise ValueError(f"Expected json string at position {node.char_position}, got {node.type}")

    return node


class JsonParser:
    @staticmethod
    async def parse(stream: Tokenator) -> ParsingContext:
        root = JsonRoot()
        task = create_task(JsonParser._parse_root(root, stream))
        return ParsingContext(root, task)

    @staticmethod
    async def _parse_root(root: JsonRoot, stream: Tokenator):
        try:
            node_resolver = RootNodeResolver()
            await root.parse(stream, node_resolver)
            node = await root.node()
            if isinstance(node, ComplexNode):
                await node.parse(stream, node_resolver)
        finally:
            await JsonParser._drain_stream(stream)

    @staticmethod
    async def _drain_stream(stream: Tokenator):
        async for _ in stream:
            pass
