#!/usr/bin/env python3
import logging.config
import os
from pathlib import Path

from aidial_sdk import DIALApp
from starlette.responses import Response

from aidial_assistant.application.assistant_application import (
    AssistantApplication,
)

# Get the log level from the environment variable
log_level = os.getenv("LOG_LEVEL") or "INFO"

config_dir = Path(os.getenv("CONFIG_DIR") or "aidial_assistant/configs")

# Load the logging configuration
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s: %(asctime)s %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }
)
app = DIALApp()
app.add_chat_completion("assistant", AssistantApplication(config_dir))


@app.get("/healthcheck/status200")
def status200() -> Response:
    return Response("Service is running...", status_code=200)
