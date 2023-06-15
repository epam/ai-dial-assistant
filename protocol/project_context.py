import json
from typing import Dict, Callable, List

from protocol.commands.base import Command

CommandDict = Dict[str, Callable[[Dict], Command]]


class ProjectContext:
    def __init__(self, command_dict: CommandDict):
        self.command_dict = command_dict

    def create_command(self, dict: Dict) -> Command:
        if "command" not in dict:
            raise Exception(
                f"Can't find 'command' key in the Json describing a command: {json.dumps(dict)}"
            )

        command = dict["command"]

        if command not in self.command_dict.keys():
            raise Exception(
                f"The command '{command}' is expected to be one of {self.command_dict.keys()}"
            )
        else:
            return self.command_dict[command](dict)

    def parse_commands(self, command_str: str) -> List[Command]:
        invocation = json.loads(command_str)
        return list(map(lambda d: self.create_command(d), invocation["commands"]))
