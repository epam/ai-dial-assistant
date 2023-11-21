import logging
from functools import wraps

from aidial_sdk import HTTPException
from openai import OpenAIError

logger = logging.getLogger(__name__)


def openai_error_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except OpenAIError as e:
            logger.exception("Unhandled OpenAI error")
            if e.error:
                raise HTTPException(
                    message=e.error.message,
                    status_code=e.http_status or 500,
                    type=e.error.type,
                    code=e.error.code,
                    param=e.error.param,
                )

            raise

    return wrapper
