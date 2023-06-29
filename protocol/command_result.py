import json
from enum import Enum
from typing import Any, List, TypedDict

from protocol.commands.base import Command, ExecutionCallback
from utils.printing import print_exception


class Status(str, Enum):
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class CommandResult(TypedDict):
    status: Status
    response: str
    """Response provided by the human.
            Contains both result of a successful command execution and
            error messages for the failed one."""


def responses_to_text(responses: List[CommandResult]) -> str:
    return json.dumps({"responses": responses})


async def execute_command(command: Command, args: List[Any], execution_callback: ExecutionCallback) -> CommandResult:
    response: Any
    try:
        response = await command.execute(args, execution_callback)
        status = Status.SUCCESS
    except Exception as e:
        print_exception()
        response = str(e)
        status = Status.ERROR

    return {"status": status, "response": response}
