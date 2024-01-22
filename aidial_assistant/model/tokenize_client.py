from urllib.parse import urljoin

from pydantic import BaseModel

from aidial_assistant.model.model_client import ModelClientRequest
from aidial_assistant.utils.requests import apost


class TokenizeClientRequest(BaseModel):
    inputs: list[ModelClientRequest | str]


class TokenizeClient:
    def __init__(self, base_url: str):
        self.url = urljoin(base_url, "tokenize")

    async def tokenize(self, request: TokenizeClientRequest) -> list[int, str]:
        async with apost(self.url, request) as response:
            return response.json()
