from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel, PositiveInt, parse_obj_as

from aidial_assistant.utils.yaml_loader import Loader


class OpenAIConf(BaseModel):
    model: str
    temperature: float
    request_timeout: int
    api_base: str


class ChatConf(BaseModel):
    buffer_size: PositiveInt


T = TypeVar("T")


def read_conf(type_: Type[T], path: Path) -> T:
    data = yaml.load(path.open(), Loader=Loader)
    return parse_obj_as(type_, data)
