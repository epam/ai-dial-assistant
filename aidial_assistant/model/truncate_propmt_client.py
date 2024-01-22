from urllib.parse import urljoin

from pydantic import BaseModel

from aidial_assistant.model.model_client import ModelClientRequest
from aidial_assistant.utils.requests import apost


class TruncatePromptClientRequest(BaseModel):
    inputs: list[ModelClientRequest]


class TruncatePromptClient:
    def __init__(self, api_base: str):
        self.url = urljoin(api_base, "truncate_prompt")

    async def truncate_prompt(
        self, request: TruncatePromptClientRequest
    ) -> list[list[int] | str]:
        async with apost(self.url, request) as response:
            return response["outputs"]
