import httpx
import pytest
from aidial_sdk import HTTPException
from openai import OpenAIError, APIStatusError

from aidial_assistant.utils.exceptions import (
    RequestParameterValidationError,
    unhandled_exception_handler,
)

ERROR_MESSAGE = "<error message>"
PARAM = "<param>"


@pytest.mark.asyncio
async def test_request_parameter_validation_error():
    @unhandled_exception_handler
    async def function():
        raise RequestParameterValidationError(ERROR_MESSAGE, PARAM)

    with pytest.raises(HTTPException) as exc_info:
        await function()

    assert (
        repr(exc_info.value)
        == f"HTTPException(message='{ERROR_MESSAGE}', status_code=422,"
        f" type='invalid_request_error', param='{PARAM}', code=None)"
    )


@pytest.mark.asyncio
async def test_openai_error():
    @unhandled_exception_handler
    async def function():
        raise OpenAIError(ERROR_MESSAGE)

    with pytest.raises(HTTPException) as exc_info:
        await function()

    assert (
        repr(exc_info.value)
        == f"HTTPException(message='{ERROR_MESSAGE}', status_code=500,"
        f" type='internal_server_error', param=None, code=None)"
    )


@pytest.mark.asyncio
async def test_openai_error_with_json_body():
    http_status = 123
    error_type = "<error type>"
    error_code = "<error code>"
    json_body = {
        "type": error_type,
        "code": error_code,
        "param": PARAM,
    }

    @unhandled_exception_handler
    async def function():
        raise APIStatusError(
            ERROR_MESSAGE,
            response=httpx.Response(
                request=httpx.Request("GET", "http://localhost"),
                status_code=http_status,
            ),
            body=json_body,
        )

    with pytest.raises(HTTPException) as exc_info:
        await function()

    assert (
        repr(exc_info.value)
        == f"HTTPException(message='{ERROR_MESSAGE}', status_code={http_status},"
        f" type='{error_type}', param='{PARAM}', code='{error_code}')"
    )


@pytest.mark.asyncio
async def test_generic_exception():
    @unhandled_exception_handler
    async def function():
        raise Exception(ERROR_MESSAGE)

    with pytest.raises(HTTPException) as exc_info:
        await function()

    assert (
        repr(exc_info.value)
        == f"HTTPException(message='{ERROR_MESSAGE}', status_code=500,"
        f" type='internal_server_error', param=None, code=None)"
    )
