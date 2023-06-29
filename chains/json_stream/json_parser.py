import json
from asyncio import create_task, Queue
from collections.abc import AsyncIterator

from typing_extensions import override

from chains.json_stream.json_root import JsonRoot, RootNodeResolver
from chains.json_stream.parsing_context import ParsingContext
from chains.json_stream.json_array import JsonArray
from chains.json_stream.json_node import JsonNode, NodeResolver
from chains.json_stream.json_normalizer import JsonNormalizer
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_string import JsonString
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


async def to_string(node: JsonNode) -> AsyncIterator[str]:
    if isinstance(node, JsonString):
        yield '"'
        async for token in string_node(node):
            yield json.dumps(token)[1:-1]
        yield '"'
    elif isinstance(node, JsonObject):
        yield '{'
        separate = False
        async for key, value in object_node(node):
            if separate:
                yield ', '
            yield json.dumps(key)
            yield ': '
            async for token in to_string(value):
                yield token
            separate = True
        yield '}'
    elif isinstance(node, JsonArray):
        yield '['
        separate = False
        async for value in array_node(node):
            if separate:
                yield ','
            async for token in to_string(value):
                yield token
            separate = True
        yield ']'
    else:
        raise ValueError(f"Unexpected node type: {node.type}")


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
            await node.parse(stream, node_resolver)
        finally:
            await JsonParser._drain_stream(stream)

    @staticmethod
    async def _drain_stream(stream: Tokenator):
        async for _ in stream:
            pass
