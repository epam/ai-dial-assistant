import jinja2
from dotenv import load_dotenv

from utils.text import decapitalize


def init():
    jinja2.filters.FILTERS["decap"] = decapitalize  # type: ignore

    load_dotenv()  # take environment variables from .env.
