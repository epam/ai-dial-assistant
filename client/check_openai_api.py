import json
import os
from typing import Iterator, Generator, Any, Iterable

import openai
from openai.openai_object import OpenAIObject


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


openai.api_base = "http://localhost:5000/"
if __name__ == "__main__":
    response: Iterable[Any] = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {
                'role': 'user',
                'content': 'What is the weather tomorrow in London?'
            }
        ],
        temperature=0,
        stream=True,
    )
    total_response = [{}]
    for chunk in response:
        os.system('cls')
        print(json.dumps(total_response[0], indent=4).replace('\\n', '\n'))
        total_response = merge(total_response, chunk.to_dict_recursive()["choices"])
