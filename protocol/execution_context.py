import json
from typing import Dict, List, Iterator

from pydantic import BaseModel

from protocol.commands.base import Command, CommandConstructor

CommandDict = dict[str, CommandConstructor]


class CommandListElem(BaseModel):
    command: str

    class Config:
        extra = "allow"


class CommandList(BaseModel):
    commands: List[CommandListElem]


class ExecutionContext:
    def __init__(self, command_dict: CommandDict):
        self.command_dict = command_dict

    def create_command(self, name: str) -> Command:
        available_commands = set(self.command_dict.keys())

        if name not in available_commands:
            raise Exception(
                f"The command '{name}' is expected to be one of {available_commands}"
            )

        return self.command_dict[name]()

