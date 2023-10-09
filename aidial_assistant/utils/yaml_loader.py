import json
import os
from typing import IO, Any

import yaml

from aidial_assistant.utils.env import get_env


class Loader(yaml.SafeLoader):
    """YAML Loader with the state: root directory of the loaded file."""

    root: str

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root = os.path.split(stream.name)[0]
        except AttributeError:
            self.root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(
        os.path.join(loader.root, loader.construct_scalar(node))  # type: ignore
    )
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r") as f:
        if extension in ("yaml", "yml"):
            return yaml.load(f, Loader)
        elif extension in ("json",):
            return json.load(f)
        else:
            return "".join(f.readlines())


def construct_env(loader: Loader, node: yaml.Node) -> Any:
    """Lookup environment variable referenced at node."""

    name = str(loader.construct_yaml_str(node))
    return get_env(name)


yaml.add_constructor("!include", construct_include, Loader)
yaml.add_constructor("!env", construct_env, Loader)
