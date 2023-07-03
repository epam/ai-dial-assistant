from collections.abc import Callable, Awaitable
from typing import TypeVar

import aiohttp
from aiohttp import ClientResponse

T = TypeVar("T")


async def aget(url: str, parser: Callable[[ClientResponse], Awaitable[T]]) -> T:
    print(f"Fetching data from {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Unable to fetch data from {url}")

            return await parser(response)
