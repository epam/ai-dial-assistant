from typing_extensions import override

from aidial_assistant.json_stream.tokenator import AsyncPeekable, Tokenator


class JsonNormalizer(AsyncPeekable[str]):
    def __init__(self, stream: Tokenator):
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
            token = await self.stream.apeek()
            if str.isspace(token):
                await anext(self.stream)
                continue
            else:
                return token
