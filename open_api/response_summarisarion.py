from typing import Any
from chains.base_chain import BaseChain

from cli.main_args import parse_args
from llm.base import create_chat_from_conf
from prompts.open_api import OPEN_API_SUMMARY_MESSAGE


def summarize_response(api_response: Any, query: str) -> str:
    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)

    chain: BaseChain = BaseChain(model)
    chain.add_message(
        OPEN_API_SUMMARY_MESSAGE.format(api_response=api_response, query=query)
    )
    resp_str = chain.run().content

    return resp_str
