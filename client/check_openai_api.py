import json
import os
from typing import Any, Iterable, Dict, List

import openai


def merge_str(target, source):
    if target is None:
        return source
    else:
        return target + source


def merge_dicts(target, source):
    for key, value in source.items():
        if key in target:
            if isinstance(value, int):
                target[key] = value
            elif isinstance(value, str):
                target[key] = merge_str(target[key], value)
            else:
                merge(target[key], value)
        else:
            target[key] = value

    return target


def merge_lists(target, source):
    for i in source:
        index = i['index']

        if index < len(target):
            merge(target[index], i)
        else:
            target.append(i)

    return target


def merge(target, source):
    if isinstance(source, list):
        return merge_lists(target, source)

    if isinstance(source, dict):
        return merge_dicts(target, source)


openai.api_base = "http://localhost:8080"
if __name__ == "__main__":
    response: Iterable[Any] = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {
                'role': 'system',
                'content': "Do not make up any plugins. Just say: I don't know."
            },
            {
                'role': 'user',
                'content': 'What is the weather tomorrow in London in short?'
            },
            # {
            #     "role": "assistant",
            #     "content": "Tomorrow's weather in London is expected to be mostly clear with some clouds. Temperatures will range from 16.48\u00b0C to 29.33\u00b0C. Wind speeds will vary between 0.88 m/s and 2.79 m/s.",
            #     "custom_content": {
            #         "state": [
            #             {
            #                 "index": 0,
            #                 "role": "assistant",
            #                 "message": "{\"commands\": [{\"command\": \"weather-forecast\", \"args\": [\"London\", \"2023-06-24\"]}]}"
            #             },
            #             {
            #                 "index": 1,
            #                 "role": "user",
            #                 "message": "{\"responses\": [{\"status\": \"ERROR\", \"response\": \"The command 'weather-forecast' is expected to be one of {'big-number-multiplication', 'say-or-ask', 'run-plugin'}\"}]}\nImportant: you must reply with a valid JSON containing a list of commands. You must not use natural language."
            #             },
            #             {
            #                 "index": 2,
            #                 "role": "assistant",
            #                 "message": "{\"commands\": [{\"command\": \"run-plugin\", \"args\": [\"weather-forecast\", \"What is the weather tomorrow in London?\"]}]}"
            #             },
            #             {
            #                 "index": 3,
            #                 "role": "user",
            #                 "message": "{\"responses\": [{\"status\": \"SUCCESS\", \"response\": \"Tomorrow's weather in London:\\n\\n01:00 - Broken clouds, 18.04\\u00b0C, feels like 17.83\\u00b0C, humidity 74%, wind speed 2.51 m/s\\n04:00 - Clear sky, 16.48\\u00b0C, feels like 16.25\\u00b0C, humidity 79%, wind speed 2.29 m/s\\n07:00 - Clear sky, 17.88\\u00b0C, feels like 17.63\\u00b0C, humidity 73%, wind speed 2.68 m/s\\n10:00 - Clear sky, 23.04\\u00b0C, feels like 22.94\\u00b0C, humidity 59%, wind speed 2.7 m/s\\n13:00 - Clear sky, 27.45\\u00b0C, feels like 27.58\\u00b0C, humidity 46%, wind speed 2.79 m/s\\n16:00 - Few clouds, 29.33\\u00b0C, feels like 29.05\\u00b0C, humidity 41%, wind speed 2.5 m/s\\n19:00 - Scattered clouds, 26.46\\u00b0C, feels like 26.46\\u00b0C, humidity 52%, wind speed 0.88 m/s\\n22:00 - Broken clouds, 22.07\\u00b0C, feels like 21.85\\u00b0C, humidity 58%, wind speed 2.67 m/s\"}]}\nImportant: you must reply with a valid JSON containing a list of commands. You must not use natural language."
            #             }
            #         ]
            #     }
            # },
            # {
            #     "role": "user",
            #     "content": "Should I bring an umbrella?"
            # }
        ],
        temperature=0,
        stream=True,
        addons=[{"url": "https://www.wolframalpha.com/.well-known/ai-plugin.json"}],
    )
    total_response = [{}]
    for chunk in response:
        os.system('cls')
        total_response: List[Dict] = merge(total_response, chunk.to_dict_recursive()["choices"])
        print(json.dumps(total_response[0], indent=4).replace('\\n', '\n'))
