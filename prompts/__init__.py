import jinja2
from utils.text import decapitalize

jinja2.filters.FILTERS["decap"] = decapitalize  # type: ignore
