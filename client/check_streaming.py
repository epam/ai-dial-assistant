from langchain.schema import HumanMessage

from chains.model_client import ModelClient
from cli.main_args import parse_args
from llm.base import create_chat_from_conf

if __name__ == "__main__":
    args = parse_args()
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)
    model_client = ModelClient(model=model)
    generator = model_client.stream([HumanMessage(content="Hello")])
    for message in generator:
        print(message.content, end="")
