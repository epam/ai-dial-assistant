import json
from asyncio import Queue
from collections.abc import AsyncIterator

from typing_extensions import override

from aidial_assistant.json_stream.exceptions import unexpected_symbol_error
from aidial_assistant.json_stream.json_node import ComplexNode, NodeResolver
from aidial_assistant.json_stream.tokenator import Tokenator


class JsonString(ComplexNode[str], AsyncIterator[str]):
    def __init__(self, char_position: int):
        super().__init__(char_position)
        self._listener = Queue[str | None]()
        self._buffer = ""

    @override
    def type(self) -> str:
        return "string"

    @staticmethod
    def token() -> str:
        return '"'

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    @override
    async def __anext__(self) -> str:
        result = await self._listener.get()
        if result is None:
            raise StopAsyncIteration

        self._buffer += result
        return result

    @override
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        async for token in JsonString.read(stream):
            await self._listener.put(token)
        await self._listener.put(None)

    @override
    async def to_string_tokens(self) -> AsyncIterator[str]:
        yield JsonString.token()
        async for token in self:
            yield json.dumps(token)[1:-1]
        yield JsonString.token()

    @staticmethod
    async def read(stream: Tokenator) -> AsyncIterator[str]:
        char = await anext(stream)
        if not char == JsonString.token():
            raise unexpected_symbol_error(char, stream.char_position)
        result = ""
        token_position = stream.token_position
        while True:
            char = await anext(stream)
            if char == JsonString.token():
                break

            result += await JsonString.escape(stream) if char == "\\" else char
            if token_position != stream.token_position:
                yield result
                result = ""
                token_position = stream.token_position

        if result:
            yield result

    @staticmethod
    async def escape(stream: Tokenator) -> str:
        char = await anext(stream)
        if char == "u":
            unicode_sequence = "".join([await anext(stream) for _ in range(4)])  # type: ignore
            return str(int(unicode_sequence, 16))
        if char in '"\\/':
            return char
        if char == "b":
            return "\b"
        elif char == "f":
            return "\f"
        elif char == "n":
            return "\n"
        elif char == "r":
            return "\r"
        elif char == "t":
            return "\t"
        else:
            # Ignore when model cannot escape text properly
            return char
            # raise ValueError(f"Unexpected escape sequence: \\{char}" + " at " + str(stream.char_position - 1))

    @override
    def value(self) -> str:
        return self._buffer
