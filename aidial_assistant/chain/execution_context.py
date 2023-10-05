from aidial_assistant.commands.base import Command, CommandConstructor

CommandDict = dict[str, CommandConstructor]


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
