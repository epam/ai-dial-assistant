import asyncio

from langchain.schema import HumanMessage

from chains.model_client import ModelClient
from cli.main_args import parse_args
from llm.base import create_openai_chat
from utils.open_ai import get_openai_key


async def main():
    args = parse_args("..")
    model = create_openai_chat(args.openai_conf, get_openai_key())
    model_client = ModelClient(model=model)
    tokens = model_client.agenerate([HumanMessage(content="Hello")])
    async for token in tokens:
        print(token, end="")

if __name__ == "__main__":
    asyncio.run(main())
