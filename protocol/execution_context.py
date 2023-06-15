import json
from typing import Dict, List

from pydantic import BaseModel

from protocol.commands.base import Command, CommandConstructor

CommandDict = Dict[str, CommandConstructor]


class CommandListElem(BaseModel):
    command: str

    class Config:
        extra = "allow"


class CommandList(BaseModel):
    commands: List[CommandListElem]


class ExecutionContext:
    def __init__(self, command_dict: CommandDict):
        self.command_dict = command_dict

    def _create_command(self, cmd: CommandListElem) -> Command:
        name = cmd.command
        available_commands = set(self.command_dict.keys())

        if name not in available_commands:
            raise Exception(
                f"The command '{name}' is expected to be one of {available_commands}"
            )

        return self.command_dict[name](cmd.dict())

    def parse_commands(self, command_str: str) -> List[Command]:
        try:
            obj = json.loads(command_str)
            commands = CommandList.parse_obj(obj).commands
        except Exception as e:
            raise Exception(f"Can't parse commands: {str(e)}")
        return list(map(lambda d: self._create_command(d), commands))
