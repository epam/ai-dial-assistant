from asyncio import Queue
from collections.abc import AsyncIterator

from typing_extensions import override

from chains.json_stream.json_node import JsonNode, NodeResolver, ComplexNode, unexpected_symbol_error
from chains.json_stream.json_normalizer import JsonNormalizer
from chains.json_stream.tokenator import Tokenator


class JsonArray(ComplexNode, AsyncIterator[JsonNode]):
    def __init__(self, char_position: int):
        super().__init__(char_position)
        self.listener = Queue[JsonNode | None | BaseException]()

    @override
    def type(self) -> str:
        return 'array'

    @staticmethod
    def token() -> str:
        return '['

    @override
    def __aiter__(self) -> AsyncIterator[JsonNode]:
        return self

    @override
    async def __anext__(self) -> JsonNode:
        result = await self.listener.get()
        if result is None:
            raise StopAsyncIteration

        return ComplexNode.throw_if_exception(result)

    @override
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        try:
            normalised_stream = JsonNormalizer(stream)
            char = await anext(normalised_stream)
            self._char_position = stream.char_position
            if not char == JsonArray.token():
                raise unexpected_symbol_error(char, stream.char_position)

            separate = False
            while True:
                char = await normalised_stream.apeek()
                if char == ']':
                    await anext(normalised_stream)
                    break

                if char == ',':
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
        except BaseException as e:
            await self.listener.put(e)

    @override
    async def to_string_tokens(self) -> AsyncIterator[str]:
        yield JsonArray.token()
        separate = False
        async for value in self:
            if separate:
                yield ', '
            async for token in value.to_string_tokens():  # type: ignore
                yield token
            separate = True
        yield ']'
