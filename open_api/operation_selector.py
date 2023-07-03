import json
from typing import Union

from langchain.tools import APIOperation, OpenAPISpec
from pydantic import BaseModel

from chains.model_client import ModelClient
from cli.main_args import parse_args
from llm.base import create_openai_chat
from prompts.open_api import OPEN_API_SELECTOR_MESSAGE
from utils.open_ai import get_openai_key


class OpenAPICommand(BaseModel):
    command: str
    args: dict


class OpenAPIClarification(BaseModel):
    user_question: str


OpenAPIResponse = Union[OpenAPICommand, OpenAPIClarification]


class OpenAPIResponseWrapper(BaseModel):
    """Just a wrapper class for the union to ease parsing"""

    resp: OpenAPIResponse

    @staticmethod
    def parse_str(s) -> OpenAPIResponse:
        return OpenAPIResponseWrapper.parse_obj({"resp": json.loads(s)}).resp


OpenAPIOperations = dict[str, APIOperation]


def collect_operations(spec: OpenAPISpec, base_url: str) -> OpenAPIOperations:
    operations: dict[str, APIOperation] = {}

    def add_operation(spec, path, method):
        operation = APIOperation.from_openapi_spec(spec, path, method)
        if operation.base_url is None:
            operation.base_url = base_url
        operations[operation.operation_id] = operation

    if spec.paths is None:
        return operations

    for path, path_item in spec.paths.items():
        if path_item.get is not None:
            add_operation(spec, path, "get")
        if path_item.post is not None:
            add_operation(spec, path, "post")

    return operations


def select_open_api_operation(
    api_description: str, ops: OpenAPIOperations, query: str
) -> OpenAPIResponse:
    api_schema = "\n\n".join([op.to_typescript() for op in ops.values()])

    args = parse_args()
    model = create_openai_chat(args.openai_conf, get_openai_key())

    client = ModelClient(model=model)
    message = client.generate(
        [OPEN_API_SELECTOR_MESSAGE.format(api_description=api_description, api_schema=api_schema, query=query)]
    )

    return OpenAPIResponseWrapper.parse_str(message.content)
