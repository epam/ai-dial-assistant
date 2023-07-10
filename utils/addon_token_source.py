from aiohttp import hdrs
from starlette.datastructures import Headers


class AddonTokenSource:
    def __init__(self, headers: Headers, urls: list[str]):
        self.headers = headers
        self.urls = {url: f"addon-token-{index}" for index, url in enumerate(urls)}

    def get_token(self, url: str) -> str | None:
        return self.headers.get(self.urls[url])

    @property
    def default_auth(self) -> str | None:
        return self.headers.get(hdrs.AUTHORIZATION)
