import json
from enum import Enum
from typing import List, TypedDict, Any


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
    args: dict[str, Any]


class Commands(TypedDict):
    commands: list[CommandInvocation]


class Responses(TypedDict):
    responses: list[CommandResult]


def responses_to_text(responses: List[CommandResult]) -> str:
    return json.dumps(Responses(responses=responses))


def commands_to_text(commands: List[CommandInvocation]) -> str:
    return json.dumps(Commands(commands=commands))
