from typing import Any, List


def print_list(ls: List[Any]) -> str:
    return "[" + ", ".join(map(lambda x: str(x), ls)) + "]"
