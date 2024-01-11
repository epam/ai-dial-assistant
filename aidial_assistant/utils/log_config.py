def get_log_config(log_level: str) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s | %(asctime)s | %(name)s | %(process)d | %(message)s",
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
            "aidial_sdk": {"level": log_level},
        },
    }
