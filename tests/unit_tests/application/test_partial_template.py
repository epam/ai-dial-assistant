import pytest

from aidial_assistant.application.prompts import PartialTemplate

TEST_DATA = [
    ({"a": "a1"}, {"b": "b2"}, {"c": "c3"}, "a1b2c3"),
    ({"a": "a1"}, {"a": "a2", "b": "b2"}, {"c": "c3"}, "a2b2c3"),
    ({"a": "a1"}, {"a": "a2", "b": "b2"}, {"a": "a3", "c": "c3"}, "a3b2c3"),
    (
        {"a": "a1"},
        {"a": "a2", "b": "b2"},
        {"a": "a3", "b": "b3", "c": "c3"},
        "a3b3c3",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("init,build,render,expected", TEST_DATA)
async def test_json_parsing(
    init: dict, build: dict, render: dict, expected: str
):
    template = PartialTemplate("{{a}}{{b}}{{c}}", globals=init)

    assert template.build(**build).render(**render) == expected
