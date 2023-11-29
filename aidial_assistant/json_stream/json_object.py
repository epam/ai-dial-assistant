import json
from collections.abc import AsyncIterator
from typing import Any, Tuple

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
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.utils.text import join_string


class JsonObject(ReadableNode[dict[str, Any], Tuple[str, JsonNode]]):
    def __init__(
        self, source: AsyncIterator[Tuple[str, JsonNode]], char_position: int
    ):
        super().__init__(source, char_position)
        self._object = {}

    @override
    def type(self) -> str:
        return "object"

    @staticmethod
    def token() -> str:
        return "{"

    async def get(self, key: str) -> JsonNode:
        if key in self._object.keys():
            return self._object[key]

        async for k, v in self:
            if k == key:
                return v

        raise KeyError(key)

    @override
    async def to_string_chunks(self) -> AsyncIterator[str]:
        yield JsonObject.token()
        separate = False
        async for key, value in self:
            if separate:
                yield ", "
            yield json.dumps(key)
            yield ": "
            async for chunk in value.to_string_chunks():
                yield chunk
            separate = True
        yield "}"

    @override
    def value(self) -> dict[str, Any]:
        return {k: v.value() for k, v in self._object.items()}

    @override
    def _accumulate(self, element: Tuple[str, JsonNode]):
        self._object[element[0]] = element[1]

    @classmethod
    def parse(
        cls, stream: CharacterStream, dependency_resolver: NodeResolver
    ) -> "JsonObject":
        return cls(
            JsonObject.read(stream, dependency_resolver), stream.char_position
        )

    @staticmethod
    async def read(
        stream: CharacterStream, dependency_resolver: NodeResolver
    ) -> AsyncIterator[Tuple[str, JsonNode]]:
        try:
            normalised_stream = JsonNormalizer(stream)
            char = await anext(normalised_stream)
            if not char == JsonObject.token():
                raise unexpected_symbol_error(char, stream.char_position)

            separate = False
            while True:
                char = await normalised_stream.apeek()

                if char == "}":
                    await normalised_stream.askip()
                    break

                if char == ",":
                    if not separate:
                        raise unexpected_symbol_error(
                            char, stream.char_position
                        )

                    await normalised_stream.askip()
                    separate = False
                elif char == '"':
                    if separate:
                        raise unexpected_symbol_error(
                            char, stream.char_position
                        )

                    key = await join_string(JsonString.read(stream))
                    colon = await anext(normalised_stream)
                    if not colon == ":":
                        raise unexpected_symbol_error(
                            colon, stream.char_position
                        )

                    value = await dependency_resolver.resolve(stream)
                    yield key, value

                    if isinstance(value, ReadableNode):
                        await value.read_to_end()
                    separate = True
                else:
                    raise unexpected_symbol_error(char, stream.char_position)
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)
