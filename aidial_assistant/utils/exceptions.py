import logging
from functools import wraps

from aidial_sdk import HTTPException
from openai import APIError

logger = logging.getLogger(__name__)


class RequestParameterValidationError(Exception):
    def __init__(self, message: str, param: str, *args: object) -> None:
        super().__init__(message, *args)
        self._param = param

    @property
    def param(self) -> str:
        return self._param


def _to_http_exception(e: Exception) -> HTTPException:
    if isinstance(e, RequestParameterValidationError):
        return HTTPException(
            message=str(e),
            status_code=422,
            type="invalid_request_error",
            param=e.param,
        )

    if isinstance(e, APIError):
        raise HTTPException(
            message=e.message,
            status_code=getattr(e, "status_code", None) or 500,
            type=e.type or "runtime_error",
            code=e.code,
            param=e.param,
        )

    return HTTPException(
        message=str(e), status_code=500, type="internal_server_error"
    )


def unhandled_exception_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception("Unhandled exception")
            raise _to_http_exception(e)

    return wrapper
