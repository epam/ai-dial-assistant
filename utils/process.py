import subprocess
from pathlib import Path
from typing import List, Tuple


def run_exe(
    exe: str, args: List[str], cwd: Path, trust: bool = False
) -> Tuple[str, str, int]:
    command = [exe] + args

    confirmation = "y"
    if not trust:
        confirmation = input(
            "Do you want to continue? [Y,y] for continue, anything else for explanation of request rejection: "
        )

    if confirmation.lower() == "y":
        process = subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        return process.stdout, process.stderr, process.returncode

    raise Exception(f"Can't run the command: {confirmation}")


def print_exe_result(res: Tuple[str, str, int]) -> str:
    stdout, stderr, exitcode = res

    if exitcode == 0:
        return f"The command successfully executed with stdout:\n{stdout}"
    else:
        n = 25
        stderr_short = "\n".join(stderr.split("\n")[:n])
        raise Exception(
            f"The command failed with exitcode {exitcode} and stderr (first {n} lines):\n{stderr_short}"
        )
