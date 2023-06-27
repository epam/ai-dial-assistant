from typing import Dict, List

from typing_extensions import override

from protocol.commands.base import Command, ExecutionCallback
from utils.files import get_project_root
from utils.process import print_exe_result, run_exe


class RunPython(Command):
    @staticmethod
    def token():
        return "run-python"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 1
        self.source = dict["args"][0]

    @override
    async def execute(self, args: List[str], execution_callback: ExecutionCallback) -> str:
        assert len(args) == 1
        source = args[0]

        cwd = get_project_root() / ".tmp"
        cwd.mkdir(exist_ok=True, parents=True)

        source_file = cwd / "source.py"
        source_file.write_text(source)

        return print_exe_result(run_exe("python", [str(source_file)], cwd, trust=True))
