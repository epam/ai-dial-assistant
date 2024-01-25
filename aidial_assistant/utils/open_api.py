from typing import Any, Iterable, Tuple

from langchain_community.utilities.openapi import OpenAPISpec
from openai.types.chat import ChatCompletionToolParam
from openapi_pydantic import DataType, Reference, Schema

from aidial_assistant.utils.open_ai import construct_tool


def _resolve_schema(spec: OpenAPISpec, schema: Schema | Reference) -> Schema:
    if isinstance(schema, Reference):
        return spec.get_referenced_schema(schema)

    return schema


def _construct_property(
    spec: OpenAPISpec, schema: Schema | Reference
) -> dict[str, Any]:
    return _resolve_schema(spec, schema).dict(exclude_none=True)


def _extract_body_parameters(
    spec: OpenAPISpec, schema: Schema | Reference
) -> Iterable[Tuple[str, dict[str, Any], bool]]:
    schema = _resolve_schema(spec, schema)
    if schema.type != DataType.OBJECT:
        raise ValueError("Body schema must be an object")

    if schema.properties:
        required = schema.required or []
        for prop_name, prop_schema in schema.properties.items():
            prop_schema = _resolve_schema(spec, prop_schema)

            yield (
                prop_name,
                _construct_property(spec, prop_schema),
                prop_name in required,
            )


def construct_tool_from_spec(
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

                for name, prop, is_required in _extract_body_parameters(
                    spec, media_type.media_type_schema
                ):
                    properties[name] = prop
                    if is_required:
                        required.append(name)
                break

    operation_id = OpenAPISpec.get_cleaned_operation_id(operation, path, method)
    return construct_tool(
        operation_id, operation.description or "", properties, required
    )
