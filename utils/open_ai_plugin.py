from aiohttp import ClientResponse
from langchain.tools import OpenAPISpec
from pydantic import BaseModel, parse_obj_as

from utils.http import aget


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


async def get_open_ai_plugin_info(url: str) -> OpenAIPluginInfo:
    """Takes url pointing to .well-known/ai-plugin.json file"""
    ai_plugin = await aget(url, _parse_ai_plugin_conf)
    open_api = await aget(ai_plugin.api.url.replace("0.0.0.0", "localhost"), _parse_openapi_spec)

    return OpenAIPluginInfo(ai_plugin=ai_plugin, open_api=open_api)


async def _parse_ai_plugin_conf(response: ClientResponse) -> AIPluginConf:
    return parse_obj_as(AIPluginConf, await response.json(content_type="text/json"))


async def _parse_openapi_spec(response: ClientResponse) -> OpenAPISpec:
    return OpenAPISpec.from_text(await response.text())
