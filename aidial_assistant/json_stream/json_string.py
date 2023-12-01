import json
from collections.abc import AsyncIterator

from typing_extensions import override

from aidial_assistant.json_stream.chunked_char_stream import ChunkedCharStream
from aidial_assistant.json_stream.exceptions import (
    unexpected_end_of_stream_error,
    unexpected_symbol_error,
    JsonParsingException,
)
from aidial_assistant.json_stream.json_node import CompoundNode


class JsonString(CompoundNode[str, str]):
    def __init__(self, source: AsyncIterator[str], char_position: int):
        super().__init__(source, char_position)
        self._buffer = ""

    @override
    def type(self) -> str:
        return "string"

    @override
    def _accumulate(self, element: str):
        self._buffer += element

    @override
    async def to_string_chunks(self) -> AsyncIterator[str]:
        yield '"'
        async for chunk in self:
            yield json.dumps(chunk)[1:-1]
        yield '"'

    @override
    def value(self) -> str:
        return self._buffer

    @classmethod
    def parse(cls, stream: ChunkedCharStream) -> "JsonString":
        return cls(JsonString.read(stream), stream.char_position)

    @staticmethod
    async def read(stream: ChunkedCharStream) -> AsyncIterator[str]:
        try:
            char = await anext(stream)
            if not JsonString.starts_with(char):
                raise unexpected_symbol_error(char, stream.char_position)
            result = ""
            chunk_position = stream.chunk_position
            while True:
                char = await anext(stream)
                if char == '"':
                    break

                result += (
                    await JsonString._escape(stream) if char == "\\" else char
                )
                if chunk_position != stream.chunk_position:
                    yield result
                    result = ""
                    chunk_position = stream.chunk_position
        except StopAsyncIteration:
            raise unexpected_end_of_stream_error(stream.char_position)

        if result:
            yield result

    @staticmethod
    async def _escape(stream: ChunkedCharStream) -> str:
        char = await anext(stream)
        if char == "u":
            unicode_sequence = "".join([await anext(stream) for _ in range(4)])  # type: ignore
            return chr(int(unicode_sequence, 16))
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
            raise JsonParsingException(
                f"Unexpected escape sequence: \\{char}.",
                stream.char_position - 1,
            )

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == '"'
