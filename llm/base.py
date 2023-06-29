from typing import List, Optional

import openai
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from conf.project_conf import ChatConf, OpenAIConf
from llm.callback import CallbackWithNewLines


def create_chat_from_conf(openai_conf: OpenAIConf, chat_conf: ChatConf, openai_api_key: str) -> ChatOpenAI:
    callbacks: Optional[List[BaseCallbackHandler]] = (
        [CallbackWithNewLines()] if chat_conf.streaming else None
    )

    openai.log = openai_conf.openai_log_level

    if openai_conf.azure is not None:
        return AzureChatOpenAI(
            # callbacks=callbacks,
            verbose=True,
            streaming=chat_conf.streaming,
            model_name=openai_conf.model_name,
            temperature=openai_conf.temperature,
            request_timeout=openai_conf.request_timeout,
            **openai_conf.azure.dict(),
        )  # type: ignore

    return ChatOpenAI(
        # callbacks=callbacks,
        verbose=True,
        streaming=chat_conf.streaming,
        model_name=openai_conf.model_name,
        openai_api_key=openai_api_key,
        temperature=openai_conf.temperature,
        request_timeout=openai_conf.request_timeout,
    )  # type: ignore
