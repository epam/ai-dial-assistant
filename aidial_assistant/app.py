import logging.config
import os
from pathlib import Path

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig, TracingConfig

from aidial_assistant.utils.log_config import get_log_config

log_level = os.getenv("LOG_LEVEL", "INFO")
otlp_export_enabled: bool = (
    os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") is not None
)
config_dir = Path(os.getenv("CONFIG_DIR", "aidial_assistant/configs"))
tools_supporting_deployments: set[str] = set(
    os.getenv("TOOLS_SUPPORTING_DEPLOYMENTS", "").split(",")
)
otel_logging_enabled: bool = True


logging.config.dictConfig(get_log_config(log_level, otel_logging_enabled))

telemetry_config = TelemetryConfig(
    tracing=TracingConfig(
        otlp_export=otlp_export_enabled, logging=otel_logging_enabled
    ),
)
app = DIALApp(telemetry_config=telemetry_config)

# A delayed import is necessary to set up the httpx hook before the openai client inherits from AsyncClient.
from aidial_assistant.application.assistant_application import (  # noqa: E402
    AssistantApplication,
)

app.add_chat_completion(
    "assistant",
    AssistantApplication(config_dir, tools_supporting_deployments),
)
