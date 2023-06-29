from typing import Any

from chains.model_client import ModelClient
from cli.main_args import parse_args
from llm.base import create_chat_from_conf
from prompts.open_api import OPEN_API_SUMMARY_MESSAGE
from utils.open_ai import get_openai_key


def summarize_response(api_response: Any, query: str) -> str:
    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf, get_openai_key())

    client = ModelClient(model=model)
    message = client.generate([OPEN_API_SUMMARY_MESSAGE.format(api_response=api_response, query=query)])
    return message.content
