from collections.abc import AsyncIterator

from chains.json_stream.json_array import JsonArray
from chains.json_stream.json_node import JsonNode
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_parser import string_node, array_node, object_node
from utils.text import join_string


class CommandParser:
    def __init__(self, node: JsonNode):
        try:
            self.node = object_node(node)
        except Exception as e:
            raise Exception(f"Cannot parse command object: {e}")

    async def parse_name(self) -> str:
        try:
            return await join_string(string_node(await self.node.get("command")))
        except Exception as e:
            raise Exception(f"Cannot parse command name: {e}")

    async def parse_args(self) -> AsyncIterator[JsonNode]:
        try:
            args = await self.node.get("args")
            # HACK: model not always passes args as an array
            if isinstance(args, JsonObject):
                yield args
            else:
                async for arg in array_node(args):
                    yield arg
        except Exception as e:
            raise Exception(f"Cannot parse command args array: {e}")
