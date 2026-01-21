import pytest
from langchain_community.utilities.openapi import OpenAPISpec

from aidial_assistant.utils.open_api import construct_tool_from_spec

OPEN_API_SPEC = {
    "openapi": "3.0.2",
    "info": {
        "title": "Test API title",
        "description": "Test API description",
        "version": "0.0.1",
    },
    "servers": [{"url": ".."}],
    "paths": {
        "/path_with_parameters": {
            "get": {
                "description": "Tool with parameters",
                "operationId": "id_with_parameters",
                "parameters": [
                    {
                        "name": "param1",
                        "in": "query",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "description": "First parameter",
                        },
                    },
                    {
                        "name": "param2",
                        "in": "query",
                        "schema": {
                            "type": "string",
                            "description": "Second parameter",
                        },
                    },
                ],
            }
        },
        "/path_with_body_parameters": {
            "post": {
                "description": "Tool with body parameters",
                "operationId": "id_with_body_parameters",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/SchemaWithBodyParameters"
                            }
                        }
                    },
                },
            }
        },
    },
    "components": {
        "schemas": {
            "SchemaWithBodyParameters": {
                "required": ["param1"],
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "First parameter",
                    },
                    "param2": {
                        "type": "string",
                        "description": "Second parameter",
                    },
                },
            }
        }
    },
}

EXPECTED_TOOLS = [
    (
        "/path_with_parameters",
        "get",
        {
            "type": "function",
            "function": {
                "name": "id_with_parameters",
                "description": "Tool with parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "First parameter",
                        },
                        "param2": {
                            "type": "string",
                            "description": "Second parameter",
                        },
                    },
                    "required": ["param1"],
                },
            },
        },
    ),
    (
        "/path_with_body_parameters",
        "post",
        {
            "type": "function",
            "function": {
                "name": "id_with_body_parameters",
                "description": "Tool with body parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "First parameter",
                        },
                        "param2": {
                            "type": "string",
                            "description": "Second parameter",
                        },
                    },
                    "required": ["param1"],
                },
            },
        },
    ),
]


@pytest.mark.parametrize("path,method,expected", EXPECTED_TOOLS)
def test_construct_tool_from_spec(path, method, expected):
    actual = construct_tool_from_spec(
        OpenAPISpec.from_spec_dict(OPEN_API_SPEC), path, method
    )

    assert actual == expected
