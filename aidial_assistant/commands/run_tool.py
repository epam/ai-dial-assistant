from typing import Any

from langchain_community.tools.openapi.utils.api_models import APIOperation
from langchain_community.utilities.openapi import OpenAPISpec
from openai.types.chat import ChatCompletionToolParam
from openapi_pydantic import Reference, Schema
from typing_extensions import override

from aidial_assistant.commands.base import (
    Command,
    CommandConstructor,
    ExecutionCallback,
    ResultObject,
    TextResult,
    get_required_field,
)
from aidial_assistant.commands.open_api import OpenAPIChatCommand
from aidial_assistant.commands.plugin_callback import PluginChainCallback
from aidial_assistant.commands.run_plugin import PluginInfo
from aidial_assistant.model.model_client import (
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.tools_chain.tools_chain import CommandToolDict, ToolsChain
from aidial_assistant.utils.open_ai import (
    construct_tool,
    system_message,
    user_message,
)


def _construct_property(
    spec: OpenAPISpec, schema: Schema | Reference
) -> dict[str, Any]:
    return (
        schema
        if isinstance(schema, Schema)
        else spec.get_referenced_schema(schema)
    ).dict(exclude_none=True)


def _construct_tool(
    spec: OpenAPISpec, path: str, method: str
) -> ChatCompletionToolParam:
    operation = spec.get_operation(path, method)
    properties: dict[str, Any] = {}
    required = []
    for p in spec.get_parameters_for_operation(operation):
        if p.param_schema is None:
            raise ValueError(f"Parameter {p.name} has no schema")

        properties[p.name] = _construct_property(spec, p.param_schema)

        if p.required:
            required.append(p.name)

    request_body = spec.get_request_body_for_operation(operation)
    if request_body is not None:
        for key, media_type in request_body.content.items():
            if key == "application/json":
                if media_type.media_type_schema is None:
                    raise ValueError("Body has no schema")

                parameter_name = "body"
                properties[parameter_name] = _construct_property(
                    spec, media_type.media_type_schema
                )
                required.append(parameter_name)
                break

    operation_id = OpenAPISpec.get_cleaned_operation_id(operation, path, method)
    return construct_tool(
        operation_id, operation.description or "", properties, required
    )


class RunTool(Command):
    def __init__(
        self, model: ModelClient, plugin: PluginInfo, max_completion_tokens: int
    ):
        self.model = model
        self.plugin = plugin
        self.max_completion_tokens = max_completion_tokens

    @staticmethod
    def token():
        return "run-tool"

    @override
    async def execute(
        self, args: dict[str, Any], execution_callback: ExecutionCallback
    ) -> ResultObject:
        query = get_required_field(args, "query")

        spec = self.plugin.info.open_api
        spec_url = self.plugin.info.get_full_spec_url()

        def create_command(operation: APIOperation) -> CommandConstructor:
            # The function is necessary to capture the current value of op.
            # Otherwise, only first op will be used for all commands
            return lambda: OpenAPIChatCommand.create(
                spec_url, operation, self.plugin.auth
            )

        commands: CommandToolDict = {
            operation.operation_id: (create_command(operation), tool)
            for path in spec.paths
            for operation, tool in (
                (
                    APIOperation.from_openapi_spec(spec, path, method),
                    _construct_tool(spec, path, method),
                )
                for method in spec.get_methods_for_path(path)
            )
        }

        chain = ToolsChain(self.model, commands, self.max_completion_tokens)

        messages = [
            system_message(self.plugin.info.ai_plugin.description_for_model),
            user_message(query),
        ]
        chain_callback = PluginChainCallback(execution_callback)
        try:
            await chain.run_chat(messages, chain_callback)
        except ReasonLengthException:
            pass

        return TextResult(chain_callback.result)
