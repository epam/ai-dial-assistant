import os


def get_env(name: str, default: str | None = None) -> str:
    if name in os.environ:
        val = os.environ.get(name)
        if val is not None:
            return val

    if default:
        return default

    raise Exception(f"{name} env variable is not set")
