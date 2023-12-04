from collections.abc import AsyncIterator

from aidial_assistant.json_stream.chunked_char_stream import ChunkedCharStream
from aidial_assistant.json_stream.json_array import JsonArray
from aidial_assistant.json_stream.json_node import JsonNode
from aidial_assistant.json_stream.json_object import JsonObject
from aidial_assistant.json_stream.json_parser import (
    array_node,
    object_node,
    string_node,
)
from aidial_assistant.utils.text import join_string


class AssistantProtocolException(Exception):
    pass


async def skip_to_json_start(stream: ChunkedCharStream):
    # Some models tend to provide explanations for their replies regardless of what the prompt says.
    try:
        while True:
            char = await stream.apeek()
            if JsonObject.starts_with(char):
                break

            await anext(stream)
    except StopAsyncIteration:
        raise AssistantProtocolException("Reply must be in JSON format.")


class CommandReader:
    def __init__(self, node: JsonNode):
        try:
            self.node = object_node(node)
        except TypeError as e:
            raise AssistantProtocolException(
                f"Cannot parse command object: {e}"
            )

    async def parse_name(self) -> str:
        try:
            return await join_string(
                string_node(await self.node.get("command"))
            )
        except (TypeError, KeyError) as e:
            raise AssistantProtocolException(f"Cannot parse command name: {e}")

    async def parse_args(self) -> AsyncIterator[JsonNode]:
        try:
            args = await self.node.get("args")
            # HACK: model not always passes args as an array
            if isinstance(args, JsonArray):
                async for arg in array_node(args):
                    yield arg
            else:
                yield args
        except (TypeError, KeyError) as e:
            raise AssistantProtocolException(
                f"Cannot parse command args array: {e}"
            )


class CommandsReader:
    def __init__(self, node: JsonNode):
        try:
            self.node = object_node(node)
        except TypeError as e:
            raise AssistantProtocolException(f"Failed to parse request: {e}")

    async def parse_invocations(self) -> AsyncIterator[CommandReader]:
        commands = await self._parse_command_array()
        async for command in commands:
            yield CommandReader(command)

    async def _parse_command_array(self) -> JsonArray:
        try:
            return array_node(await self.node.get("commands"))
        except (TypeError, KeyError) as e:
            raise AssistantProtocolException(
                f"Failed to parse array of command: {e}"
            )
