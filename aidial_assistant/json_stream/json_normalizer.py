from typing_extensions import override

from aidial_assistant.json_stream.characterstream import (
    AsyncPeekable,
    CharacterStream,
)


class JsonNormalizer(AsyncPeekable[str]):
    def __init__(self, stream: CharacterStream):
        self.stream = stream

    @override
    def __aiter__(self):
        return self

    @override
    async def __anext__(self) -> str:
        await self.apeek()
        return await anext(self.stream)

    @override
    async def apeek(self) -> str:
        while True:
            char = await self.stream.apeek()
            if str.isspace(char):
                await anext(self.stream)
                continue
            else:
                return char
