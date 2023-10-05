import os
import typing

import openai

if __name__ == "__main__":
    response: typing.Iterable[typing.Any] = openai.ChatCompletion.create(
        api_base="http://localhost:5000",
        api_type="azure",
        api_version="2023-03-15-preview",
        api_key=os.environ["RAIL_PROXY_API_KEY"],
        engine="assistant",
        model="gpt-4",
        temperature=0,
        timeout=300,
        messages=[
            {
                "role": "system",
                "content": "Use 'epam-10k-semantic-search' plugin to answer questions about EPAM",
            },
            {"role": "user", "content": "What was EPAM revenue in 2022?"},
            {
                "role": "assistant",
                "content": "EPAM's revenue in 2022 was $4.825 billion.",
                "custom_content": {
                    "state": {
                        "invocations": [
                            {
                                "index": 0,
                                "request": '{"commands": [{"command": "run-plugin", "args": ["epam-10k-semantic-search", "What was EPAM revenue in 2022?"]}]}',
                                "response": '{"responses": [{"status": "SUCCESS", "response": "EPAM\'s revenue in 2022 was $4.825 billion."}]}',
                            }
                        ]
                    }
                },
            },
            {"role": "user", "content": "Is it more than in 2021?"},
        ],
        addons=[
            {
                "url": "https://epam-qna-application.staging.deltixhub.io/semantic-search/.well-known/ai-plugin.json"
            }
        ],
    )

    print(response)
