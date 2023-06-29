from collections.abc import AsyncIterator

from chains.command_parser import CommandParser
from chains.json_stream.json_array import JsonArray
from chains.json_stream.json_node import JsonNode
from chains.json_stream.json_object import JsonObject
from chains.json_stream.json_parser import string_node, array_node, object_node
from utils.text import join_string


class RequestParser:
    def __init__(self, node: JsonNode):
        try:
            self.node = object_node(node)
        except Exception as e:
            raise Exception(f"Failed to parse request: {e}")

    async def parse_invocations(self) -> AsyncIterator[CommandParser]:
        commands = await self._parse_command_array()
        async for command in commands:
            yield CommandParser(command)

    async def _parse_command_array(self) -> JsonArray:
        try:
            return array_node(await self.node.get("commands"))
        except Exception as e:
            raise Exception(f"Failed to parse array of command: {e}")
