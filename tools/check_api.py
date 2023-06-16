import os

import requests

if __name__ == "__main__":
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    response = requests.post(
        "http://localhost:5000/chat/completions",
        # "https://api.openai.com/v1/chat/completions",
        headers=headers,
        stream=True,
        json={
            "model": "gpt-4-0613",
            "messages": [
                # {"role": "system", "content": "Do not make up any function arguments if unknown."},
                {"role": "user", "content": "What is your name?"},
                # {"role": "start-plugin", "name": "w-f", "query": "What is the weather today in Malaga?"},
                # {"role": "plugin", "content": "call weather-forecast"},
                # {"role": "end-plugin", "content": "30C"},
                # {"role": "assistant", "content": "The weather in Malaga is 30C"},

                # {"role": "assistant", "content": "To provide the weather forecast, I need to know your location. Could you please tell me where you are?"},
                # {"role": "user", "content": "Malaga"},
            ],
            "temperature": 0,
            "stream": True
        })

    for line in response.iter_lines():
        # filter out keep-alive new lines
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)

    # print(response.text)