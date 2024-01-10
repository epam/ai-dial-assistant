import logging.config
import os
from pathlib import Path

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig, TracingConfig

from aidial_assistant.utils.log_config import get_log_config

log_level = os.getenv("LOG_LEVEL", "INFO")
config_dir = Path(os.getenv("CONFIG_DIR", "aidial_assistant/configs"))

logging.config.dictConfig(get_log_config(log_level))

telemetry_config = TelemetryConfig(
    service_name="aidial-assistant", tracing=TracingConfig()
)
app = DIALApp(telemetry_config=telemetry_config)

# A delayed import is necessary to set up the httpx hook before the openai client inherits from AsyncClient.
from aidial_assistant.application.assistant_application import (  # noqa: E402
    AssistantApplication,
)

app.add_chat_completion(
    "assistant",
    AssistantApplication(config_dir),
)
