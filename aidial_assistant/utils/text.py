from collections.abc import AsyncIterator


def decapitalize(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]


async def join_string(stream: AsyncIterator[str]) -> str:
    return "".join([token async for token in stream])
