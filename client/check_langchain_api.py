from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

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
        ]
    )
    # content = llm_result.generations[0][-1].text
