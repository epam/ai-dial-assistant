import json
from asyncio import Queue
from collections.abc import AsyncIterator
from typing import Any, Tuple

from typing_extensions import override

from aidial_assistant.json_stream.characterstream import CharacterStream
from aidial_assistant.json_stream.exceptions import unexpected_symbol_error
from aidial_assistant.json_stream.json_node import (
    ComplexNode,
    JsonNode,
    NodeResolver,
)
from aidial_assistant.json_stream.json_normalizer import JsonNormalizer
from aidial_assistant.json_stream.json_string import JsonString
from aidial_assistant.utils.text import join_string


class JsonObject(
    ComplexNode[dict[str, Any]], AsyncIterator[Tuple[str, JsonNode]]
):
    def __init__(self, char_position: int):
        super().__init__(char_position)
        self.listener = Queue[Tuple[str, JsonNode] | None]()
        self._object: dict[str, JsonNode] = {}

    @override
    def type(self) -> str:
        return "object"

    def __aiter__(self) -> AsyncIterator[Tuple[str, JsonNode]]:
        return self

    @override
    async def __anext__(self) -> Tuple[str, JsonNode]:
        result = await self.listener.get()
        if result is None:
            raise StopAsyncIteration

        self._object[result[0]] = result[1]
        return result

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
    async def parse(
        self, stream: CharacterStream, dependency_resolver: NodeResolver
    ):
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
                    raise unexpected_symbol_error(char, stream.char_position)

                await normalised_stream.askip()
                separate = False
            elif char == '"':
                if separate:
                    raise unexpected_symbol_error(char, stream.char_position)

                key = await join_string(JsonString.read(stream))
                colon = await anext(normalised_stream)
                if not colon == ":":
                    raise unexpected_symbol_error(colon, stream.char_position)

                value = await dependency_resolver.resolve(stream)
                await self.listener.put((key, value))
                if isinstance(value, ComplexNode):
                    await value.parse(stream, dependency_resolver)
                separate = True
            else:
                raise unexpected_symbol_error(char, stream.char_position)

        await self.listener.put(None)

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
