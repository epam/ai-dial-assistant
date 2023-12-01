import json
import random
from collections.abc import AsyncIterator

import pytest

from aidial_assistant.json_stream.chunked_char_stream import ChunkedCharStream
from aidial_assistant.json_stream.exceptions import JsonParsingException
from aidial_assistant.json_stream.json_parser import (
    object_node,
    parse_json,
    string_node,
)
from aidial_assistant.utils.text import join_string

JSON_STRINGS = [
    """
    {
      "name": "John",
      "age": 30,
      "city": "New York"
    }
    """,
    """
    {
      "employees": [
        {"firstName": "John", "lastName": "Doe"},
        {"firstName": "Anna", "lastName": "Smith"},
        {"firstName": "Peter", "lastName": "Jones"}
      ]
    }
    """,
    """
    {
      "name": "John",
      "age": 30,
      "cars": {
        "car1": "Ford",
        "car2": "BMW",
        "car3": "Fiat"
      }
    }
    """,
    """
    {
      "name": "John",
      "age": 30,
      "isMarried": true,
      "hobbies": ["Reading", "Cycling", "Hiking"],
      "address": {
        "street": "123 Main St",
        "city": "New York",
        "postalCode": "10001"
      },
      "children": null
    }
    """,
    """
    {
      "students": [
        {
          "id": 1,
          "name": "John",
          "grade": "A"
        },
        {
          "id": 2,
          "name": "Anna",
          "grade": "B"
        },
        {
          "id": 3,
          "name": "Peter",
          "grade": "C"
        }
      ]
    }
    """,
    """
    {}
    """,
    """
    {
      "text": "Hello, World!\\nThis is a test.\\tTabbed."
    }
    """,
    """
    {
      "text": "Hello, \\u4e16\\u754c"
    }
    """,
    """
    {
      "isActive": true,
      "isDeleted": false,
      "middleName": null
    }
    """,
    """
    {
      "integer": 10,
      "negativeInteger": -5,
      "float": 20.5,
      "negativeFloat": -10.2,
      "zero": 0,
      "largeNumber": 1234567890
    }
    """,
    """
    {
      "arrayOfIntegers": [1, 2, 3, 4, 5],
      "arrayOfFloats": [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    """,
    """
    {
      "nestedNumbers": {
        "positive": {
          "integer": 10,
          "float": 20.5
        },
        "negative": {
          "integer": -5,
          "float": -10.2
        }
      }
    }
    """,
    """
    {
      "objectWithNumbers": [
        {"id": 1, "value": 10.5},
        {"id": 2, "value": 20.5},
        {"id": 3, "value": 30.5}
      ]
    }
    """,
]


async def _split_into_chunks(json_string: str) -> AsyncIterator[str]:
    while json_string:
        chunk_size = random.randint(
            1, 5
        )  # generate a random size between 1 and 5
        chunk_size = min(
            chunk_size, len(json_string)
        )  # make sure we don't exceed the length of string
        chunk, json_string = json_string[:chunk_size], json_string[chunk_size:]
        yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("json_string", JSON_STRINGS)
async def test_json_parsing(json_string: str):
    node = await parse_json(ChunkedCharStream(_split_into_chunks(json_string)))
    actual = await join_string(node.to_string_chunks())
    expected = json.dumps(json.loads(json_string))

    assert actual == expected


@pytest.mark.asyncio
async def test_incomplete_json_parsing():
    incomplete_json_string = """
    {
      "test": "field"
    """
    node = object_node(
        await parse_json(
            ChunkedCharStream(_split_into_chunks(incomplete_json_string))
        )
    )
    _, value = await anext(node)
    await string_node(value).read_to_end()

    assert node.value() == {"test": "field"}


@pytest.mark.asyncio
async def test_incorrect_escape_sequence():
    incomplete_json_string = '"\\k"'
    node = string_node(
        await parse_json(
            ChunkedCharStream(_split_into_chunks(incomplete_json_string))
        )
    )

    with pytest.raises(JsonParsingException) as exc_info:
        await node.read_to_end()

    assert str(exc_info.value) == (
        "Failed to parse json string at position 2: Unexpected escape sequence: \\k."
    )
