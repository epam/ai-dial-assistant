from queue import Queue
from typing import Optional, Any
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from typing_extensions import override

from llm.callback import CallbackWithNewLines


if __name__ == "__main__":
    callbacks = [CallbackWithNewLines()]
    model = ChatOpenAI(
        streaming=True,
        callbacks=callbacks,
        model_name="gpt-4",
        openai_api_base="http://localhost:5000",
        verbose=True,
        temperature=0,
        request_timeout=60,
    )  # type: ignore

    llm_result = model.generate(
        [
            [
                HumanMessage(
                    content="What is the weather tomorrow in London in short?"
                ),
                AIMessage(
                    content="Tomorrow's weather in London will be mostly clear with some clouds. Temperatures will range from 16.48°C to 29.33°C, and wind speeds will vary between 0.88 m/s and 2.79 m/s."
                ),
                HumanMessage(content="Should I bring an umbrella?"),
            ]
        ],
    )
    # content = llm_result.generations[0][-1].text
