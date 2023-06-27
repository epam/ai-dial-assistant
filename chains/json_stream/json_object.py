from asyncio import Queue
from collections.abc import AsyncIterator
from typing import Tuple, Dict, Any

from typing_extensions import override

from chains.json_stream.json_node import JsonNode, NodeResolver
from chains.json_stream.json_normalizer import JsonNormalizer
from chains.json_stream.json_string import JsonString
from chains.json_stream.tokenator import Tokenator
from utils.text import join_string


class JsonObject(JsonNode, AsyncIterator[Tuple[str, JsonNode]]):
    def __init__(self):
        self.listener = Queue[Tuple[str, JsonNode] | None | Exception]()
        self.object: Dict[str, Any] = {}

    def __aiter__(self) -> AsyncIterator[Tuple[str, JsonNode]]:
        return self

    @override
    async def __anext__(self) -> Tuple[str, JsonNode]:
        result = await self.listener.get()
        if result is None:
            raise StopAsyncIteration

        return JsonNode.throw_if_exception(result)

    @staticmethod
    def token() -> str:
        return '{'

    async def get(self, key: str) -> JsonNode:
        if key in self.object.keys():
            return self.object[key]

        async for k, v in self:
            self.object[k] = v
            if k == key:
                return v

        raise KeyError(key)

    @override
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        try:
            normalised_stream = JsonNormalizer(stream)
            char = await anext(normalised_stream)
            if not char == JsonObject.token():
                raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")

            separate = False
            while True:
                char = await normalised_stream.apeek()

                if char == '}':
                    await normalised_stream.askip()
                    break

                if char == ',':
                    if not separate:
                        raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")

                    await normalised_stream.askip()
                    separate = False
                elif char == '"':
                    if separate:
                        raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")

                    key = await join_string(JsonString.read(stream))
                    colon = await anext(normalised_stream)
                    if not colon == ':':
                        raise Exception(f"Unexpected symbol: {colon} at {stream.char_position}")

                    value = await dependency_resolver.resolve(stream)
                    await self.listener.put((key, value))
                    await value.parse(stream, dependency_resolver)
                    separate = True
                else:
                    raise Exception(f"Unexpected symbol: {char} at {stream.char_position}")

            await self.listener.put(None)
        except Exception as e:
            await self.listener.put(e)
