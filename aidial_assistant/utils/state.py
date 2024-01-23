from typing import TypedDict


class Invocation(TypedDict):
    index: str | int
    request: str
    response: str


class State(TypedDict, total=False):
    invocations: list[Invocation]
