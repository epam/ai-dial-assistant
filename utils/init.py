import jinja2

from utils.text import decapitalize


def init():
    jinja2.filters.FILTERS["decap"] = decapitalize  # type: ignore
