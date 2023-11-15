from asyncio import Queue
from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from aidial_assistant.json_stream.characterstream import CharacterStream
from aidial_assistant.json_stream.exceptions import unexpected_symbol_error
from aidial_assistant.json_stream.json_node import (
    ComplexNode,
    JsonNode,
    NodeResolver,
)
from aidial_assistant.json_stream.json_normalizer import JsonNormalizer


class JsonArray(ComplexNode[list[Any]], AsyncIterator[JsonNode]):
    def __init__(self, char_position: int):
        super().__init__(char_position)
        self.listener = Queue[JsonNode | None]()
        self.array: list[JsonNode] = []

    @override
    def type(self) -> str:
        return "array"

    @staticmethod
    def token() -> str:
        return "["

    @override
    def __aiter__(self) -> AsyncIterator[JsonNode]:
        return self

    @override
    async def __anext__(self) -> JsonNode:
        result = await self.listener.get()
        if result is None:
            raise StopAsyncIteration

        self.array.append(result)
        return result

    @override
    async def parse(
        self, stream: CharacterStream, dependency_resolver: NodeResolver
    ):
        normalised_stream = JsonNormalizer(stream)
        char = await anext(normalised_stream)
        self._char_position = stream.char_position
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
                    raise unexpected_symbol_error(char, stream.char_position)

                await anext(normalised_stream)
                separate = False
            else:
                value = await dependency_resolver.resolve(stream)
                await self.listener.put(value)
                if isinstance(value, ComplexNode):
                    await value.parse(stream, dependency_resolver)
                separate = True

        await self.listener.put(None)

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
        return [item.value() for item in self.array]
