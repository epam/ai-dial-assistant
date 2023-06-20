from typing import List

from colorama import Fore, Style
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.callback import CallbackWithNewLines

hist1 = [
    HumanMessage(content="What is the weather tomorrow in London in short?"),
    AIMessage(
        content=""">>>Run command: weather-forecast("London", "2023-06-17")<<<
>>>The command 'weather-forecast' is expected to be one of {'run-plugin', 'say-or-ask', 'big-number-multiplication'}<<<
>>>Run command: run-plugin("weather-forecast", "What is the weather tomorrow in London?")<<<
>>>Tomorrow's weather in London:

01:00 - Broken clouds, 19.3°C, feels like 18.56°C, 49% humidity, 2.68 m/s wind
04:00 - Broken clouds, 14.75°C, feels like 13.85°C, 60% humidity, 2.26 m/s wind
07:00 - Broken clouds, 16.23°C, feels like 15.37°C, 56% humidity, 2.67 m/s wind
10:00 - Overcast clouds, 20.56°C, feels like 19.77°C, 42% humidity, 2.68 m/s wind
13:00 - Light rain, 23.29°C, feels like 22.75°C, 41% humidity, 3.08 m/s wind
16:00 - Overcast clouds, 24.87°C, feels like 24.64°C, 47% humidity, 3.5 m/s wind
19:00 - Light rain, 20.71°C, feels like 20.64°C, 69% humidity, 3.6 m/s wind
22:00 - Light rain, 16.46°C, feels like 16.43°C, 87% humidity, 2.46 m/s wind<<<
Tomorrow's weather in London is expected to be mostly cloudy with occasional light rain. Temperatures will range from 14.75°C to 24.87°C, with wind speeds between 2.26 and 3.6 m/s."""
    ),
    HumanMessage(content="Should I bring an umbrella?"),
]

hist3: List[BaseMessage] = [
    HumanMessage(content="What is the temperature tomorrow in London?"),
]

hist2: List[BaseMessage] = [HumanMessage(content="What is your name?")]

if __name__ == "__main__":
    callbacks = [CallbackWithNewLines()]
    model = ChatOpenAI(
        streaming=True,
        callbacks=callbacks,
        model_name="gpt-4",
        openai_api_base="http://127.0.0.1:5000",
        verbose=True,
        temperature=0,
        request_timeout=6000,
    )  # type: ignore

    history: List[BaseMessage] = []

    while True:
        llm_result = model.generate([history])

        content = llm_result.generations[0][-1].text
        message = AIMessage(content=content)
        history.append(message)

        print(Fore.RED, flush=True)
        content = input("\n> ")
        print(Style.RESET_ALL, flush=True)

        message = HumanMessage(content=content)
        history.append(message)
