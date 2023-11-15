from typing import AsyncIterator


async def to_async_string(string: str) -> AsyncIterator[str]:
    yield string


def to_async_strings(sequence: list[str]) -> list[AsyncIterator[str]]:
    return [to_async_string(item) for item in sequence]


def to_async_repeated_string(
    string: str, count: int
) -> list[AsyncIterator[str]]:
    return [to_async_string(string) for _ in range(count)]