from pydantic import BaseModel, parse_obj_as
import requests
from langchain.tools import OpenAPISpec


class AuthConf(BaseModel):
    type: str


class ApiConf(BaseModel):
    type: str
    url: str
    has_user_authentication: bool


class AIPluginConf(BaseModel):
    schema_version: str
    name_for_model: str
    name_for_human: str
    description_for_model: str
    description_for_human: str
    auth: AuthConf
    api: ApiConf
    logo_url: str
    contact_email: str
    legal_info_url: str


class OpenAIPluginInfo(BaseModel):
    ai_plugin: AIPluginConf
    open_api: OpenAPISpec


def get_open_ai_plugin_info(url: str) -> OpenAIPluginInfo:
    """Takes url pointing to .well-known/ai-plugin.json file"""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Unable to fetch data from {url}")

    ai_plugin = parse_obj_as(AIPluginConf, response.json())
    open_api = OpenAPISpec.from_url(ai_plugin.api.url)

    return OpenAIPluginInfo(ai_plugin=ai_plugin, open_api=open_api)
