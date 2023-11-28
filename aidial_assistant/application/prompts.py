from datetime import datetime

from jinja2 import Environment, Template

from aidial_assistant.utils.text import decapitalize

JINJA2_ENV = Environment()
JINJA2_ENV.filters["decap"] = decapitalize


class DateAwareTemplate(Template):
    def render(self, *args, **kwargs):
        # Compute today's date in the format "DD Mmm YYYY"
        today = datetime.now().strftime("%d %b %Y")
        return super().render(*args, **kwargs, today_date=today)


class PartialTemplate:
    def __init__(self, template: str, **kwargs):
        self.template = template
        self.template_class_args = kwargs

    def build(self, **kwargs) -> Template:
        template_args = self.template_class_args.get("globals", {}) | kwargs
        return JINJA2_ENV.from_string(
            self.template,
            **(self.template_class_args | {"globals": template_args}),
        )


_REQUEST_EXAMPLE_TEXT = """
You should ALWAYS reply with a JSON containing an array of commands:
{
  "commands": [
    {
      "command": "<command name>",
      "args": [
        // <array of arguments>
      ]
    }
  ]
}
The commands are invoked by system on user's behalf.
""".strip()

_SYSTEM_TEXT = """
Today's date is {{today_date}}.
This message defines the following communication protocol.

{%- if system_prefix %}
{{system_prefix}}
{%- endif %}

# Protocol
{{request_response}}

## Commands
{%- if tools %}
> run-plugin
The command runs a specified plugin to solve a one-shot task written in natural language.
Plugins do not see current conversation and require all details to be provided in the query to solve the task.
Arguments:
 - NAME is one of the following plugins:
{%- for name, description in tools.items() %}
    * {{name}} - {{description | decap}}
{%- endfor %}
 - QUERY is the query string.
{%- endif %}
> reply
The command delivers ultimate result to the user.
Arguments:
 - MESSAGE is a string containing the final and complete result for the user.

## Example
{"commands": [{"command": "reply", "args": ["Hello, world!"]}]}
""".strip()

_PLUGIN_SYSTEM_TEXT = """
Today's date is {{today_date}}.
This message defines the following communication protocol.

# Service
API_DESCRIPTION:
{{api_description}}

# API Schema
```typescript
{{api_schema}}}
```

# Protocol
{{request_response}}

## Commands
{%- for command_name in command_names %}
> {{command_name}}
Arguments:
 - <JSON dict according to the API Schema>
{%- endfor %}
> reply
The command delivers ultimate result to the user
Arguments:
 - MESSAGE is a string containing the final and complete result for the user.

## Example
{"commands": [{"command": "reply", "args": ["Hello, world!"]}]}
""".strip()

_ENFORCE_JSON_FORMAT_TEXT = """
{{response}}
""".strip()

_MAIN_BEST_EFFORT_TEXT = (
    """
You were allowed to use the following addons to answer the query below.

=== ADDONS ===
{% for name, description in tools.items() %}
* {{name}} - {{description | decap}}
{%- endfor %}

=== QUERY ===

{{message}}

{%- if dialogue %}

=== ADDONS REQUESTS AND RESPONSES ===
{% for message in dialogue %}
{{message["role"]}}: {{message["content"]}}
{%- endfor %}
{%- endif %}

However, the follow-up requests failed with the following error:
> {{error}}

Please respond to the query using the available information, and explaining that the use of the addons was not possible due to the error.
"""
).strip()

_PLUGIN_BEST_EFFORT_TEXT = (
    """
You were allowed to use the following API to answer the query below.

=== API ===

```typescript
{{api_schema}}
```

=== QUERY ===

{{message}}

{%- if dialogue %}

=== API REQUESTS AND RESPONSES ===
{% for message in dialogue %}
{{message["role"]}}: {{message["content"]}}
{%- endfor %}
{%- endif %}

However, the follow-up requests failed with the following error:
> {{error}}

Please respond to the query using the available information, and explaining that the use of the API was not possible due to the error.
"""
).strip()

MAIN_SYSTEM_DIALOG_MESSAGE = PartialTemplate(
    _SYSTEM_TEXT,
    globals={"request_response": _REQUEST_EXAMPLE_TEXT},
    template_class=DateAwareTemplate,
)

PLUGIN_SYSTEM_DIALOG_MESSAGE = PartialTemplate(
    _PLUGIN_SYSTEM_TEXT,
    globals={"request_response": _REQUEST_EXAMPLE_TEXT},
    template_class=DateAwareTemplate,
)

ENFORCE_JSON_FORMAT_TEMPLATE = JINJA2_ENV.from_string(_ENFORCE_JSON_FORMAT_TEXT)

MAIN_BEST_EFFORT_TEMPLATE = PartialTemplate(_MAIN_BEST_EFFORT_TEXT)

PLUGIN_BEST_EFFORT_TEMPLATE = PartialTemplate(_PLUGIN_BEST_EFFORT_TEXT)
