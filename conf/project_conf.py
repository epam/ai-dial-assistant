from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, Field, parse_obj_as, root_validator

from protocol.commands.base import CommandConstructor, resolve_constructor
from utils.yaml_loader import Loader


class LogLevel(str, Enum):
    INFO = "info"
    DEBUG = "debug"


class OpenAIConf(BaseModel):
    model_name: str = "gpt-4-0314"
    temperature: float = 0.0
    request_timeout: int = 10
    openai_api_base: str
    buffer_size: int


class ChatConf(BaseModel):
    streaming: bool = True


class CommandConf(BaseModel):
    implementation: CommandConstructor  # A string <model_name>::<class_name> is parsed to a CommandConstructor on construction
    description: str
    args: List[str]
    result: str

    @root_validator(pre=True)
    def trim_strings(cls, values):
        for field in ["description", "result"]:
            if field in values:
                values[field] = values[field].strip()
        return values

    @root_validator(pre=True)
    def parse_implementation(cls, values):
        if "implementation" in values:
            impl = values["implementation"]
            if isinstance(impl, str):
                values["implementation"] = resolve_constructor(impl)
        return values


class PluginCommand(BaseModel):
    type: Literal["command"]


class PluginTool(BaseModel):
    type: Literal["tool"]
    system_prefix: str = ""
    description: str
    commands: List[str]

    @root_validator(pre=True)
    def trim_strings(cls, values):
        for field in ["description", "system_prefix"]:
            if field in values:
                values[field] = values[field].strip()
        return values


class PluginOpenAI(BaseModel):
    type: Literal["open-ai-plugin"]
    system_prefix: str = ""
    url: str

    @root_validator(pre=True)
    def trim_strings(cls, values):
        for field in ["system_prefix"]:
            if field in values:
                values[field] = values[field].strip()
        return values


PluginConfUnion = PluginCommand | PluginTool | PluginOpenAI
PluginConf = Annotated[PluginConfUnion, Field(discriminator="type")]


class Conf(BaseModel):
    system_prefix: str = ""
    commands: dict[str, CommandConf]
    plugins: dict[str, PluginConf]

    @root_validator(pre=True)
    def trim_strings(cls, values):
        for field in ["system_prefix"]:
            if field in values:
                values[field] = values[field].strip()
        return values

    @root_validator
    def check_plugin_commands(cls, values):
        commands = values.get("commands")
        plugins = values.get("plugins")

        if not (commands and plugins):
            return values

        available_commands = set(commands.keys())

        for plugin in plugins.values():
            if not isinstance(plugin, PluginTool):
                continue
            for command in plugin.commands:
                if command not in available_commands:
                    raise ValueError(
                        f"Unknown command: {command}. Available commands: {available_commands}"
                    )

        return values


T = TypeVar("T")


def read_conf(type_: Type[T], path: Path) -> T:
    data = yaml.load(path.open(), Loader=Loader)
    return parse_obj_as(type_, data)
