from utils.env import get_env


def get_openai_key() -> str:
    return get_env("OPENAI_API_KEY")
