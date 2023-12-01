from collections.abc import AsyncIterator
from typing import Any

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


class JsonArray(CompoundNode[list[Any], JsonNode]):
    def __init__(self, source: AsyncIterator[JsonNode], pos: int):
        super().__init__(source, pos)
        self._array: list[JsonNode] = []

    @override
    def type(self) -> str:
        return "array"

    @staticmethod
    async def read(
        stream: ChunkedCharStream, node_parser: NodeParser
    ) -> AsyncIterator[JsonNode]:
        try:
            await skip_whitespaces(stream)
            char = await anext(stream)
            if not JsonArray.starts_with(char):
                raise unexpected_symbol_error(char, stream.char_position)

            is_comma_expected = False
            while True:
                await skip_whitespaces(stream)
                char = await stream.apeek()
                if char == "]":
                    await stream.askip()
                    break

                if char == ",":
                    if not is_comma_expected:
                        raise unexpected_symbol_error(
                            char, stream.char_position
                        )

                    await stream.askip()
                    is_comma_expected = False
                else:
                    value = await node_parser.parse(stream)
                    yield value

                    if isinstance(value, CompoundNode):
                        await value.read_to_end()
                    is_comma_expected = True
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)

    @override
    async def to_chunks(self) -> AsyncIterator[str]:
        yield "["
        is_first_element = True
        async for value in self:
            if not is_first_element:
                yield ", "
            async for chunk in value.to_chunks():
                yield chunk
            is_first_element = False
        yield "]"

    @override
    def value(self) -> list[JsonNode]:
        return [item.value() for item in self._array]

    @override
    def _accumulate(self, element: JsonNode):
        self._array.append(element)

    @classmethod
    def parse(
        cls, stream: ChunkedCharStream, node_parser: NodeParser
    ) -> "JsonArray":
        return cls(JsonArray.read(stream, node_parser), stream.char_position)

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == "["
