import json
from typing import Union

from langchain.tools import APIOperation, OpenAPISpec
from pydantic import BaseModel

from chains.base_chain import BaseChain
from cli.main_args import parse_args
from llm.base import create_chat_from_conf
from prompts.open_api import OPEN_API_SELECTOR_MESSAGE


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


def collect_operations(spec: OpenAPISpec) -> OpenAPIOperations:
    operations: dict[str, APIOperation] = {}

    def add_operation(spec, path, method):
        operation = APIOperation.from_openapi_spec(spec, path, method)
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
    model = create_chat_from_conf(args.openai_conf, args.chat_conf)

    chain: BaseChain = BaseChain(model, "INNER:open_api_endpoint_selection")
    chain.add_message(
        OPEN_API_SELECTOR_MESSAGE.format(
            api_description=api_description, api_schema=api_schema, query=query
        )
    )
    resp_str = chain.run().content

    return OpenAPIResponseWrapper.parse_str(resp_str)
