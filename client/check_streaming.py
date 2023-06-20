from langchain.schema import HumanMessage

from chains.model_client import ModelClient
from llm.base import create_chat

if __name__ == "__main__":
    model = create_chat(
        model_name="gpt-4",
        temperature=0,
        request_timeout=60,
        streaming=True,

    )
    model_client = ModelClient(model=model)
    generator = model_client.stream([HumanMessage(content="Hello")])
    for message in generator:
        print(message.content, end="")
