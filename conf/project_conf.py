from pathlib import Path
from typing import Annotated, Literal, Type, TypeVar, List

import yaml
from pydantic import BaseModel, Field, parse_obj_as, root_validator
from protocol.commands.base import CommandConstructor

from utils.yaml_loader import Loader


class OpenAIConf(BaseModel):
    model_name: str = "gpt-4-0314"
    temperature: float = 0.0
    request_timeout: int = 10
    openai_log_level: str = "info"


class ChatConf(BaseModel):
    streaming: bool = True


class CommandConf(BaseModel):
    implementation: str | CommandConstructor
    description: str
    args: List[str]
    result: str

    @root_validator(pre=True)
    def trim_strings(cls, values):
        for field in ["description", "result"]:
            if field in values:
                values[field] = values[field].strip()
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
        for field in ["description"]:
            if field in values:
                values[field] = values[field].strip()
        return values


class PluginOpenAI(BaseModel):
    type: Literal["open-ai-plugin"]
    system_prefix: str = ""
    url: str


PluginConfUnion = PluginCommand | PluginTool | PluginOpenAI
PluginConf = Annotated[PluginConfUnion, Field(discriminator="type")]


class Conf(BaseModel):
    commands: dict[str, CommandConf]
    plugins: dict[str, PluginConf]


T = TypeVar("T")


def read_conf(type_: Type[T], path: Path) -> T:
    data = yaml.load(path.open(), Loader=Loader)
    return parse_obj_as(type_, data)
