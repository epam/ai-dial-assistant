import json
from enum import Enum
from typing import List, TypedDict


class Status(str, Enum):
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class CommandResult(TypedDict):
    status: Status
    response: str
    """Response provided by the human.
            Contains both result of a successful command execution and
            error messages for the failed one."""


class CommandInvocation(TypedDict):
    command: str
    args: list[str]


def responses_to_text(responses: List[CommandResult]) -> str:
    return json.dumps({"responses": responses})


def commands_to_text(commands: List[CommandInvocation]) -> str:
    return json.dumps({"commands": commands})
