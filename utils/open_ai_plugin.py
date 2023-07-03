from urllib.parse import urljoin

from langchain.requests import Requests
from langchain.tools import OpenAPISpec
from pydantic import BaseModel, parse_obj_as


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
    requests = Requests()
    print(f"Fetching plugin data from {url}")
    ai_plugin = await _parse_ai_plugin_conf(requests, url)
    # Resolve relative url
    ai_plugin.api.url = urljoin(url, ai_plugin.api.url)
    print(f"Fetching plugin spec from {ai_plugin.api.url}")
    open_api = await _parse_openapi_spec(requests, ai_plugin.api.url)

    return OpenAIPluginInfo(ai_plugin=ai_plugin, open_api=open_api)


async def _parse_ai_plugin_conf(requests: Requests, url: str) -> AIPluginConf:
    async with requests.aget(url) as response:
        # content_type=None to disable validation, sometimes response comes as text/json
        return parse_obj_as(AIPluginConf, await response.json(content_type=None))


async def _parse_openapi_spec(requests: Requests, url: str) -> OpenAPISpec:
    async with requests.aget(url) as response:
        return OpenAPISpec.from_text(await response.text())
