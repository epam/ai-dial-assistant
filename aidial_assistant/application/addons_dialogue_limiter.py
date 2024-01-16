from typing_extensions import override

from aidial_assistant.chain.command_chain import (
    LimitExceededException,
    ModelRequestLimiter,
)
from aidial_assistant.model.model_client import (
    ChatCompletionMessageParam,
    ModelClient,
)


class AddonsDialogueLimiter(ModelRequestLimiter):
    def __init__(self, max_dialogue_tokens: int, model_client: ModelClient):
        self.max_dialogue_tokens = max_dialogue_tokens
        self.model_client = model_client

        self._dialogue_tokens = 0
        self._initial_tokens: int | None = None

    @override
    async def verify_limit(self, messages: list[ChatCompletionMessageParam]):
        if self._initial_tokens is None:
            self._initial_tokens = await self.model_client.count_tokens(
                messages
            )
            return

        self._dialogue_tokens = (
            await self.model_client.count_tokens(messages)
            - self._initial_tokens
        )

        if self._dialogue_tokens > self.max_dialogue_tokens:
            raise LimitExceededException(
                f"Addons dialogue limit exceeded. Max tokens: {self.max_dialogue_tokens},"
                f" actual tokens: {self._dialogue_tokens}."
            )
