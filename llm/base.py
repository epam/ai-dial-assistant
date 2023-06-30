import openai
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from conf.project_conf import OpenAIConf


def create_openai_chat(openai_conf: OpenAIConf, openai_api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        # callbacks=callbacks,
        verbose=True,
        streaming=True,
        model_name=openai_conf.model_name,
        openai_api_key=openai_api_key,
        temperature=openai_conf.temperature,
        request_timeout=openai_conf.request_timeout,
    )  # type: ignore


def create_azure_chat(openai_conf: OpenAIConf, deployment_name: str, openai_api_key: str) -> ChatOpenAI:
    # callbacks: Optional[List[BaseCallbackHandler]] = (
    #     [CallbackWithNewLines()] if chat_conf.streaming else None
    # )

    openai.log = openai_conf.openai_log_level

    return AzureChatOpenAI(
        # callbacks=callbacks,
        verbose=True,
        streaming=True,
        model_name=openai_conf.model_name,
        temperature=openai_conf.temperature,
        request_timeout=openai_conf.request_timeout,
        openai_api_key=openai_api_key,
        deployment_name=deployment_name,
        **openai_conf.azure.dict(),
    )  # type: ignore

