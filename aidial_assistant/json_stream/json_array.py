from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from aidial_assistant.json_stream.characterstream import CharacterStream
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
    unexpected_symbol_error,
)
from aidial_assistant.json_stream.json_node import (
    JsonNode,
    NodeResolver,
    ReadableNode,
)
from aidial_assistant.json_stream.json_normalizer import JsonNormalizer


class JsonArray(ReadableNode[list[Any], JsonNode]):
    def __init__(self, source: AsyncIterator[JsonNode], char_position: int):
        super().__init__(source, char_position)
        self._array: list[JsonNode] = []

    @override
    def type(self) -> str:
        return "array"

    @staticmethod
    def token() -> str:
        return "["

    @override
    async def to_string_chunks(self) -> AsyncIterator[str]:
        yield JsonArray.token()
        separate = False
        async for value in self:
            if separate:
                yield ", "
            async for chunk in value.to_string_chunks():
                yield chunk
            separate = True
        yield "]"

    @override
    def value(self) -> list[JsonNode]:
        return [item.value() for item in self._array]

    @override
    def _accumulate(self, element: JsonNode):
        self._array.append(element)

    @classmethod
    def parse(
        cls, stream: CharacterStream, dependency_resolver: NodeResolver
    ) -> "JsonArray":
        return cls(
            JsonArray.read(stream, dependency_resolver), stream.char_position
        )

    @staticmethod
    async def read(
        stream: CharacterStream, dependency_resolver: NodeResolver
    ) -> AsyncIterator[JsonNode]:
        try:
            normalised_stream = JsonNormalizer(stream)
            char = await anext(normalised_stream)
            if not char == JsonArray.token():
                raise unexpected_symbol_error(char, stream.char_position)

            separate = False
            while True:
                char = await normalised_stream.apeek()
                if char == "]":
                    await anext(normalised_stream)
                    break

                if char == ",":
                    if not separate:
                        raise unexpected_symbol_error(
                            char, stream.char_position
                        )

                    await anext(normalised_stream)
                    separate = False
                else:
                    value = await dependency_resolver.resolve(stream)
                    yield value

                    if isinstance(value, ReadableNode):
                        await value.read_to_end()
                    separate = True
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)
