import os

from langchain.chat_models import ChatOpenAI

from conf.project_conf import ChatConf, OpenAIConf
from llm.callback import CallbackWithNewLines


def get_openai_key() -> str:
    key_name = "OPENAI_API_KEY"
    if key_name in os.environ:
        val = os.environ.get(key_name)
        if val is not None:
            return val

    raise Exception(f"{key_name} env variable is not set")


def create_chat(
    model_name: str, temperature: float, request_timeout: int, streaming: bool
) -> ChatOpenAI:
    callbacks = [CallbackWithNewLines()] if streaming else None
    return ChatOpenAI(
        streaming=streaming,
        callbacks=callbacks,
        model_name=model_name,
        openai_api_key=get_openai_key(),
        verbose=True,
        temperature=temperature,
        request_timeout=request_timeout,
    )  # type: ignore


def create_chat_from_conf(openai_conf: OpenAIConf, chat_conf: ChatConf) -> ChatOpenAI:
    return create_chat(
        model_name=openai_conf.model_name,
        temperature=openai_conf.temperature,
        request_timeout=openai_conf.request_timeout,
        streaming=chat_conf.streaming,
    )
