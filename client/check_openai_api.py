import json
import os
import typing

import openai

from client.utils import merge

openai.api_key = os.environ["RAIL_PROXY_API_KEY"]
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = "https://assistant-service.staging.deltixhub.io"
# openai.api_base = "http://localhost:8080"
if __name__ == "__main__":
    response: typing.Iterable[typing.Any] = openai.ChatCompletion.create(
        engine='assistant',
        model='gpt-4',
        messages=[
            {
                'role': 'system',
                'content': "Use 'epam-10k-semantic-search' plugin to answer questions about EPAM"
            },
            {
                'role': 'user',
                'content': 'What was EPAM revenue in 2022?'
            },
            {
                "role": "assistant",
                "content": "EPAM's revenue in 2022 was $4.825 billion.",
                "custom_content": {
                    "state": {
                        "invocations": [
                            {
                                "index": 0,
                                "request": "{\"commands\": [{\"command\": \"run-plugin\", \"args\": [\"epam-10k-semantic-search\", \"What was EPAM revenue in 2022?\"]}]}",
                                "response": "{\"responses\": [{\"status\": \"SUCCESS\", \"response\": \"EPAM's revenue in 2022 was $4.825 billion.\"}]}"
                            }
                        ]
                    }
                }
            },
            {
                "role": "user",
                "content": "Is it more than in 2021?"
            }
        ],
        temperature=0,
        stream=True,
        addons=[
            {
                "url": "http://backend.epam10k:5000/.well-known/ai-plugin.json"
            }
        ],
    )

    total_response = [{}]
    for chunk in response:
        os.system('cls')
        total_response: list[dict] = merge(total_response, chunk.to_dict_recursive()["choices"])
        print(json.dumps(total_response[0], indent=4))
