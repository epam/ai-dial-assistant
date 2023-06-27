from asyncio import Queue
from collections.abc import AsyncIterator

from typing_extensions import override

from chains.json_stream.json_node import JsonNode, NodeResolver
from chains.json_stream.json_normalizer import JsonNormalizer
from chains.json_stream.tokenator import Tokenator


class JsonArray(JsonNode, AsyncIterator[JsonNode]):
    def __init__(self):
        self.listener = Queue[JsonNode | None | Exception]()

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

        return JsonNode.throw_if_exception(result)

    @override
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        try:
            normalised_stream = JsonNormalizer(stream)
            char = await anext(normalised_stream)
            if not char == JsonArray.token():
                raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")

            separate = False
            while True:
                char = await normalised_stream.apeek()
                if char == ']':
                    await anext(normalised_stream)
                    break

                if char == ',':
                    if not separate:
                        raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")

                    await anext(normalised_stream)
                    separate = False
                else:
                    value = await dependency_resolver.resolve(stream)
                    await self.listener.put(value)
                    await value.parse(stream, dependency_resolver)
                    separate = True

            await self.listener.put(None)
        except Exception as e:
            await self.listener.put(e)
