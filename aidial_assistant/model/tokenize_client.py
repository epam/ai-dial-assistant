from typing import Any
from urllib.parse import urljoin

from pydantic import BaseModel

from aidial_assistant.model.model_client import ModelClientRequest
from aidial_assistant.utils.requests import apost


def _read_output(output: dict[str, Any]) -> int | str:
    status = output["status"]
    if status == "success":
        return output["token_count"]

    if status == "error":
        return output["error"]

    raise ValueError(f"Unknown status: {status}")


class TokenizeClientRequest(BaseModel):
    inputs: list[ModelClientRequest | str]


class TokenizeClient:
    def __init__(self, base_url: str):
        self.url = urljoin(base_url, "tokenize")

    async def tokenize(self, request: TokenizeClientRequest) -> list[int | str]:
        async with apost(self.url, request) as response:
            data = await response.json()
            return [_read_output(output) for output in data["outputs"]]
