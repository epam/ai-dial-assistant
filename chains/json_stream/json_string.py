import json
from asyncio import Queue
from collections.abc import AsyncIterator

from typing_extensions import override

from chains.json_stream.json_node import NodeResolver, ComplexNode, unexpected_symbol_error
from chains.json_stream.tokenator import Tokenator


class JsonString(ComplexNode, AsyncIterator[str]):
    def __init__(self, char_position: int):
        super().__init__(char_position)
        self.listener = Queue[str | None | BaseException]()

    @override
    def type(self) -> str:
        return 'string'

    @staticmethod
    def token() -> str:
        return '"'

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    @override
    async def __anext__(self) -> str:
        result = await self.listener.get()
        if result is None:
            raise StopAsyncIteration

        return ComplexNode.throw_if_exception(result)

    @override
    async def parse(self, stream: Tokenator, dependency_resolver: NodeResolver):
        try:
            async for token in JsonString.read(stream):
                await self.listener.put(token)
            await self.listener.put(None)
        except BaseException as e:
            await self.listener.put(e)

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

            result += await JsonString.escape(stream) if char == '\\' else char
            if token_position != stream.token_position:
                yield result
                result = ""
                token_position = stream.token_position

        if result:
            yield result

    @staticmethod
    async def escape(stream: Tokenator) -> str:
        char = await anext(stream)
        if char == 'u':
            unicode_sequence = ''.join([await anext(stream) for _ in range(4)])
            return str(int(unicode_sequence, 16))
        if char in '"\\/':
            return char
        if char == 'b':
            return '\b'
        elif char == 'f':
            return '\f'
        elif char == 'n':
            return '\n'
        elif char == 'r':
            return '\r'
        elif char == 't':
            return '\t'
        else:
            raise ValueError(f"Unexpected escape sequence: \\{char}" + " at " + str(stream.char_position - 1))