import os

import requests

if __name__ == "__main__":
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }

    response = requests.post(
        "http://localhost:5000/chat/completions",
        # "https://api.openai.com/v1/chat/completions",
        headers=headers,
        stream=True,
        json={
            "model": "gpt-4-0613",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather tomorrow in London?"
                },
                # {
                #     "role": "assistant",
                #     "content": "The weather will be rainy.",
                #     "custom_content": {
                #         "state": {
                #             "messages": [
                #                 {
                #                     "index": 0,
                #                     "role": "assistant",
                #                     "content": ""
                #                     #
                #                     #     [
                #                     #     {
                #                     #         "index": 0,
                #                     #         "command": "weather-forecast",
                #                     #         "args": ["What is the weather tomorrow in London?"],
                #                     #         "response": {
                #                     #             "status": "SUCCESS",
                #                     #             "content": "The weather will be rainy."
                #                     #         }
                #                     #     }
                #                     # ]
                #                 }
                #             ]
                #         }
                #     }
                # },
                # {
                #     "role": "user",
                #     "content": "Should I bring an umbrella?"
                # }
            ],
            "temperature": 0,
            "stream": True,
        },
    )

    for line in response.iter_lines():
        # filter out keep-alive new lines
        if line:
            decoded_line = line.decode("utf-8")
            print(decoded_line)

    # print(response.text)
