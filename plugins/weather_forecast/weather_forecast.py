import sys
from typing import Dict

from typing_extensions import override

from protocol.commands.base import Command
from utils.files import get_project_root
from utils.process import print_exe_result, run_exe


class WeatherForecast(Command):
    location: str
    date: str

    @staticmethod
    def token():
        return "weather-forecast"

    def __init__(self, dict: Dict):
        self.dict = dict
        assert "args" in dict and isinstance(dict["args"], list)
        assert len(dict["args"]) == 2
        self.location = dict["args"][0]
        self.date = dict["args"][1]

    @override
    def execute(self) -> str:
        cwd = get_project_root()
        script = cwd / "tools" / "get_weather_report.py"

        if not script.exists():
            raise Exception(f"Script '{script}' does not exist")

        return print_exe_result(
            run_exe(
                sys.executable,
                [str(script), "--date", self.date, "--location", self.location],
                cwd,
                trust=True,
            )
        )
