#!/usr/bin/env python3
import logging.config
import os
from pathlib import Path

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig, TracingConfig
from starlette.responses import Response

from aidial_assistant.application.assistant_application import (
    AssistantApplication,
)
from aidial_assistant.utils.log_config import get_log_config

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
CONFIG_DIR: str = Path(os.getenv("CONFIG_DIR", "aidial_assistant/configs"))
OTLP_EXPORT_ENABLED: bool = (
    os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") is not None
)

logging.config.dictConfig(get_log_config(LOG_LEVEL))

telemetry_config = TelemetryConfig(
    service_name="aidial-assistant", tracing=TracingConfig(oltp_export=OTLP_EXPORT_ENABLED)
)
app = DIALApp(telemetry_config=telemetry_config)
app.add_chat_completion("assistant", AssistantApplication(CONFIG_DIR))


@app.get("/healthcheck/status200")
def status200() -> Response:
    return Response("Service is running...", status_code=200)
