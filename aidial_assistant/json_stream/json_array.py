from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from aidial_assistant.json_stream.chunked_char_stream import ChunkedCharStream
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
    unexpected_symbol_error,
)
from aidial_assistant.json_stream.json_node import (
    CompoundNode,
    JsonNode,
    NodeResolver,
)


class JsonArray(CompoundNode[list[Any], JsonNode]):
    def __init__(self, source: AsyncIterator[JsonNode], char_position: int):
        super().__init__(source, char_position)
        self._array: list[JsonNode] = []

    @override
    def type(self) -> str:
        return "array"

    @staticmethod
    async def read(
        stream: ChunkedCharStream, node_resolver: NodeResolver
    ) -> AsyncIterator[JsonNode]:
        try:
            char = await anext(await stream.skip_whitespaces())
            if not JsonArray.starts_with(char):
                raise unexpected_symbol_error(char, stream.char_position)

            is_comma_expected = False
            while True:
                char = await (await stream.skip_whitespaces()).apeek()
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
                    value = await node_resolver.resolve(stream)
                    yield value

                    if isinstance(value, CompoundNode):
                        await value.read_to_end()
                    is_comma_expected = True
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)

    @override
    async def to_string_chunks(self) -> AsyncIterator[str]:
        yield "["
        is_comma_expected = False
        async for value in self:
            if is_comma_expected:
                yield ", "
            async for chunk in value.to_string_chunks():
                yield chunk
            is_comma_expected = True
        yield "]"

    @override
    def value(self) -> list[JsonNode]:
        return [item.value() for item in self._array]

    @override
    def _accumulate(self, element: JsonNode):
        self._array.append(element)

    @classmethod
    def parse(
        cls, stream: ChunkedCharStream, node_resolver: NodeResolver
    ) -> "JsonArray":
        return cls(JsonArray.read(stream, node_resolver), stream.char_position)

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == "["
