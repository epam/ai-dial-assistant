import logging
from typing import Iterable, Mapping
from urllib.parse import urljoin

from aiocache import cached
from aiohttp import hdrs
from langchain.tools import OpenAPISpec
from pydantic import BaseModel, parse_obj_as

from aidial_assistant.utils.requests import aget

logger = logging.getLogger(__name__)


class AuthConf(BaseModel):
    type: str
    authorization_type: str = "bearer"


class ApiConf(BaseModel):
    type: str
    url: str
    has_user_authentication: bool = False
    is_user_authenticated: bool = False


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


class AddonTokenSource:
    def __init__(self, headers: Mapping[str, str], urls: Iterable[str]):
        self.headers = headers
        self.urls = {
            url: f"x-addon-token-{index}" for index, url in enumerate(urls)
        }

    def get_token(self, url: str) -> str | None:
        return self.headers.get(self.urls[url])

    @property
    def default_auth(self) -> str | None:
        return self.headers.get(hdrs.AUTHORIZATION)


async def get_open_ai_plugin_info(addon_url: str) -> OpenAIPluginInfo:
    """Takes url pointing to .well-known/ai-plugin.json file"""
    logger.info(f"Fetching plugin info from {addon_url}")
    ai_plugin = await _parse_ai_plugin_conf(addon_url)
    # Resolve relative url
    ai_plugin.api.url = urljoin(addon_url, ai_plugin.api.url)
    logger.info(f"Fetching plugin spec from {ai_plugin.api.url}")
    open_api = await _parse_openapi_spec(ai_plugin.api.url)

    return OpenAIPluginInfo(ai_plugin=ai_plugin, open_api=open_api)


@cached()
async def _parse_ai_plugin_conf(url: str) -> AIPluginConf:
    async with aget(url) as response:
        # content_type=None to disable validation, sometimes response comes as text/json
        return parse_obj_as(
            AIPluginConf, await response.json(content_type=None)
        )


@cached()
async def _parse_openapi_spec(url: str) -> OpenAPISpec:
    async with aget(url) as response:
        return OpenAPISpec.from_text(await response.text())  # type: ignore
