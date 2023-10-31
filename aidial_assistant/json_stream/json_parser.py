from asyncio import TaskGroup
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from aidial_assistant.json_stream.exceptions import JsonParsingException
from aidial_assistant.json_stream.json_array import JsonArray
from aidial_assistant.json_stream.json_node import ComplexNode, JsonNode
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_root import JsonRoot, RootNodeResolver
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.json_stream.tokenator import Tokenator


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


class JsonParser:
    @staticmethod
    @asynccontextmanager
    async def parse(stream: Tokenator) -> AsyncGenerator[JsonNode, Any]:
        root = JsonRoot()
        try:
            async with TaskGroup() as tg:
                task = tg.create_task(JsonParser._parse_root(root, stream))
                try:
                    yield await root.node()
                finally:
                    await task
        except ExceptionGroup as e:
            raise e.exceptions[0]

    @staticmethod
    async def _parse_root(root: JsonRoot, stream: Tokenator):
        try:
            node_resolver = RootNodeResolver()
            await root.parse(stream, node_resolver)
            node = await root.node()
            if isinstance(node, ComplexNode):
                await node.parse(stream, node_resolver)
        except StopAsyncIteration:
            raise JsonParsingException(
                "Failed to parse json: unexpected end of stream."
            )
        finally:
            # flush the stream
            async for _ in stream:
                pass
