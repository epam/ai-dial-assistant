from typing import Dict, List, NamedTuple, Optional

from aiohttp import ClientResponse
from langchain.requests import Requests
from langchain.tools.openapi.utils.api_models import APIOperation
from pydantic import Field
from requests import Response


class _ParamMapping(NamedTuple):
    """Mapping from parameter name to parameter value."""

    query_params: List[str]
    body_params: List[str]
    path_params: List[str]


class OpenAPIEndpointRequester:
    """Chain interacts with an OpenAPI endpoint using natural language.
    Based on OpenAPIEndpointChain from LangChain.
    """

    operation: APIOperation
    param_mapping: _ParamMapping = Field(alias="param_mapping")

    def __init__(self, operation: APIOperation):
        self.operation = operation
        self.param_mapping = _ParamMapping(
            query_params=operation.query_params,
            body_params=operation.body_params,
            path_params=operation.path_params,
        )

    def _construct_path(self, args: Dict[str, str]) -> str:
        """Construct the path from the deserialized input."""
        path = self.operation.base_url + self.operation.path
        for param in self.param_mapping.path_params:
            path = path.replace(f"{{{param}}}", str(args.pop(param, "")))
        return path

    def _extract_query_params(self, args: Dict[str, str]) -> Dict[str, str]:
        """Extract the query params from the deserialized input."""
        query_params = {}
        for param in self.param_mapping.query_params:
            if param in args:
                query_params[param] = args.pop(param)
        return query_params

    def _extract_body_params(self, args: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Extract the request body params from the deserialized input."""
        body_params = None
        if self.param_mapping.body_params:
            body_params = {}
            for param in self.param_mapping.body_params:
                if param in args:
                    body_params[param] = args.pop(param)
        return body_params

    def deserialize_json_input(self, args: dict) -> dict:
        """Use the serialized typescript dictionary.

        Resolve the path, query params dict, and optional requestBody dict.
        """
        path = self._construct_path(args)
        body_params = self._extract_body_params(args)
        query_params = self._extract_query_params(args)
        return {
            "url": path,
            "data": body_params,
            "params": query_params,
        }

    async def execute(
        self,
        args: dict,
    ) -> dict:
        request_args = self.deserialize_json_input(args)
        # "a" for async methods
        method = getattr(Requests(), "a" + self.operation.method.value)
        print(f"Request args: {request_args}")
        async with method(**request_args) as response:
            if response.status != 200:
                method_str = str(self.operation.method.value)
                return {
                    "reason": response.reason,
                    "status_code": response.status,
                    "method:": method_str.upper(),
                    "url": request_args["url"],
                    "params": request_args["params"],
                }

            # content_type=None to disable validation, sometimes response comes as text/json
            return await response.json(content_type=None)
