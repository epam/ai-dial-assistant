import json
from collections.abc import AsyncIterator
from typing import Any, Tuple

from typing_extensions import override

from aidial_assistant.json_stream.chunked_char_stream import (
    ChunkedCharStream,
    skip_whitespaces,
)
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
    unexpected_symbol_error,
)
from aidial_assistant.json_stream.json_node import (
    CompoundNode,
    JsonNode,
    NodeParser,
)
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.utils.text import join_string


class JsonObject(CompoundNode[dict[str, Any], Tuple[str, JsonNode]]):
    def __init__(self, source: AsyncIterator[Tuple[str, JsonNode]], pos: int):
        super().__init__(source, pos)
        self._object = {}

    @override
    def type(self) -> str:
        return "object"

    async def get(self, key: str) -> JsonNode:
        if key in self._object.keys():
            return self._object[key]

        async for k, v in self:
            if k == key:
                return v

        raise KeyError(key)

    @staticmethod
    async def read(
        stream: ChunkedCharStream, node_parser: NodeParser
    ) -> AsyncIterator[Tuple[str, JsonNode]]:
        try:
            await skip_whitespaces(stream)
            char = await anext(stream)
            if not JsonObject.starts_with(char):
                raise unexpected_symbol_error(char, stream.char_position)

            is_comma_expected = False
            while True:
                await skip_whitespaces(stream)
                char = await stream.apeek()

                if char == "}":
                    await stream.askip()
                    break

                if char == ",":
                    if not is_comma_expected:
                        raise unexpected_symbol_error(
                            char, stream.char_position
                        )

                    await stream.askip()
                    is_comma_expected = False
                elif JsonString.starts_with(char):
                    if is_comma_expected:
                        raise unexpected_symbol_error(
                            char, stream.char_position
                        )

                    key = await join_string(JsonString.read(stream))
                    await skip_whitespaces(stream)
                    colon = await anext(stream)
                    if not colon == ":":
                        raise unexpected_symbol_error(
                            colon, stream.char_position
                        )

                    value = await node_parser.parse(stream)
                    yield key, value

                    if isinstance(value, CompoundNode):
                        await value.read_to_end()
                    is_comma_expected = True
                else:
                    raise unexpected_symbol_error(char, stream.char_position)
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)

    @override
    async def to_chunks(self) -> AsyncIterator[str]:
        yield "{"
        is_first_entry = True
        async for key, value in self:
            if not is_first_entry:
                yield ", "
            yield json.dumps(key)
            yield ": "
            async for chunk in value.to_chunks():
                yield chunk
            is_first_entry = False
        yield "}"

    @override
    def value(self) -> dict[str, Any]:
        return {k: v.value() for k, v in self._object.items()}

    @override
    def _accumulate(self, element: Tuple[str, JsonNode]):
        self._object[element[0]] = element[1]

    @classmethod
    def parse(
        cls, stream: ChunkedCharStream, node_parser: NodeParser
    ) -> "JsonObject":
        return cls(JsonObject.read(stream, node_parser), stream.char_position)

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == "{"
