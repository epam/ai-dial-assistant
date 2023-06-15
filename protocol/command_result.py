from enum import Enum
from typing import TypedDict

from protocol.commands.base import Command
from utils.printing import print_exception


class Status(str, Enum):
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class CommandResultDict(TypedDict):
    id: int
    status: Status
    response: str


class CommandResult:
    command: Command
    """The original command requested by the model"""

    response: str
    """Response provided by the human.
    Contains both result of a successful command execution and
    error messages for the failed one."""

    id: int

    status: Status
    """Has the execution of the command failed?
    It may happen if the command is incorrect in some way.
    E.g. references a non-existing class or method"""

    kwargs: dict = dict()
    """Extra field specific for each command"""

    def __init__(
        self,
        command: Command,
        response: str,
        response_id: int,
        status: Status,
        **kwargs
    ):
        self.command = command
        self.response = response
        self.id = response_id
        self.status = status
        self.kwargs = kwargs

    def to_dict(self) -> CommandResultDict:
        return CommandResultDict(id=self.id, status=self.status, response=self.response)


def execute_command(command: Command, response_id: int) -> CommandResult:
    response: str
    try:
        response = command.execute()
        status = Status.SUCCESS
    except Exception as e:
        print_exception()
        response = str(e)
        status = Status.ERROR

    return CommandResult(
        command=command, response=response, response_id=response_id, status=status
    )
