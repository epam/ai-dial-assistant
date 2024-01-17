def get_log_config(log_level: str, telemetry_logging: bool) -> dict:
    telemetry_fmt = (
        "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - "
        if telemetry_logging
        else ""
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": f"%(levelprefix)s | %(asctime)s | %(name)s | %(process)d | {telemetry_fmt}%(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
            },
        },
        "root": {
            "handlers": ["default"],
            "level": log_level,
        },
        "loggers": {
            "aidial_sdk": {
                "handlers": ["default"],
                "level": log_level,
            },
        },
    }
