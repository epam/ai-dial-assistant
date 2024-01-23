from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from aiohttp import ClientResponse, ClientSession


@asynccontextmanager
async def arequest(
    method: str, url: str, headers, **kwargs
) -> AsyncIterator[ClientResponse]:
    async with ClientSession(headers=headers) as session:
        async with session.request(method, url, **kwargs) as response:
            yield response


# Cannot use Requests.aget(...) from langchain because of a bug: https://github.com/langchain-ai/langchain/issues/7953
@asynccontextmanager
async def aget(url: str, headers=None) -> AsyncIterator[ClientResponse]:
    async with arequest("GET", url, headers) as response:
        yield response


@asynccontextmanager
async def apost(
    url: str, data: Any, headers=None
) -> AsyncIterator[ClientResponse]:
    async with arequest("POST", url, headers, data=data) as response:
        yield response
