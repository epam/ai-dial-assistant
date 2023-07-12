from typing import Any

import openai
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from conf.project_conf import OpenAIConf, LogLevel


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


def create_azure_chat(args: dict[str, Any], openai_api_key: str) -> ChatOpenAI:
    # callbacks = [CallbackWithNewLines()]

    openai.log = LogLevel.INFO

    args = {
        "verbose": True,
        "streaming": True,
        "openai_api_key": openai_api_key,
        "deployment_name": args["model_name"]
    } | args

    return AzureChatOpenAI(**args)  # type: ignore

