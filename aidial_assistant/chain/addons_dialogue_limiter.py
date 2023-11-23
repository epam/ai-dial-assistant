from aidial_assistant.chain.history import History, enforce_json_format
from aidial_assistant.model.model_client import ModelClient, Message


class DialogueLimitExceededException(Exception):
    pass


class AddonsDialogueLimiter:
    def __init__(
        self, prompt_tokens: int, max_tokens: int, model_client: ModelClient
    ):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.model_client = model_client

        self._total_tokens = 0

    async def verify_limit(self, messages: list[Message]):
        self._total_tokens = (
            await self.model_client.count_tokens(messages) - self.prompt_tokens
        )

        if self._total_tokens > self.max_tokens:
            raise DialogueLimitExceededException(
                f"Addons dialogue limit exceeded. Max tokens: {self.max_tokens}, actual tokens: {self._total_tokens}."
            )

    @classmethod
    async def create(
        cls,
        history: History,
        model_client: ModelClient,
        max_addons_dialogue_tokens,
    ) -> "AddonsDialogueLimiter":
        return cls(
            prompt_tokens=await model_client.count_tokens(
                enforce_json_format(
                    history.to_protocol_messages_with_system_message()
                )
            ),
            max_tokens=max_addons_dialogue_tokens,
            model_client=model_client,
        )
