import sys
from typing import Dict, Any, List

from typing_extensions import override

from protocol.commands.base import Command, ExecutionCallback, ResultObject, TextResult
from utils.files import get_project_root
from utils.process import print_exe_result, run_exe


class WeatherForecast(Command):
    @staticmethod
    def token():
        return "weather-forecast"

    @override
    async def execute(self, args: List[Any], execution_callback: ExecutionCallback) -> ResultObject:
        self.assert_arg_count(args, 2)
        location = args[0]
        date = args[1]

        cwd = get_project_root()
        script = cwd / "tools" / "get_weather_report.py"

        if not script.exists():
            raise Exception(f"Script '{script}' does not exist")

        return TextResult(print_exe_result(
            run_exe(
                sys.executable,
                [str(script), "--date", date, "--location", location],
                cwd,
                trust=True,
            ))
        )
