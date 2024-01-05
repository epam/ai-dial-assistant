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


_REQUEST_FORMAT_TEXT = """
You should ALWAYS reply with a JSON containing an array of commands:
{
  "commands": [
    {
      "command": "<command name>",
      "arguments": {
        "<arg_name>": "<arg_value>"
      }
    }
  ]
}
The commands are invoked by system on user's behalf.
""".strip()

_PROTOCOL_FOOTER = """
* reply
The command delivers final response to the user.
Arguments:
 - <message> is a string containing the final and complete result for the user.

Your goal is to answer user questions. Use relevant commands when they help to achieve the goal.

## Example
{"commands": [{"command": "reply", "arguments": {"message": "Hello, world!"}}]}
""".strip()

_SYSTEM_TEXT = """
Today's date is {{today_date}}.
This message defines the following communication protocol.

{%- if system_prefix %}
{{system_prefix}}
{%- endif %}

# Protocol
{{request_format}}

## Commands
{%- if addons %}
* run-addon
This command executes a specified addon to address a one-time task described in natural language.
Addons do not see current conversation and require all details to be provided in the query to solve the task.
Arguments:
 - <name> is one of the following addons:
{%- for name, description in addons.items() %}
    * {{name}} - {{description | decap}}
{%- endfor %}
 - <query> is the query string.
{%- endif %}
{{protocol_footer}}
""".strip()

_ADDON_SYSTEM_TEXT = """
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
{{request_format}}

## Commands
{%- for command_name in command_names %}
* {{command_name}}
Arguments:
 - <JSON dict according to the API Schema>
{%- endfor %}
{{protocol_footer}}
""".strip()

_ENFORCE_JSON_FORMAT_TEXT = """
{{response}}

**Protocol reminder: reply with commands**
""".strip()

_MAIN_BEST_EFFORT_TEXT = (
    """
You were allowed to use the following addons to answer the query below.

=== ADDONS ===
{% for name, description in addons.items() %}
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

_ADDON_BEST_EFFORT_TEXT = (
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
    globals={
        "request_format": _REQUEST_FORMAT_TEXT,
        "protocol_footer": _PROTOCOL_FOOTER,
    },
    template_class=DateAwareTemplate,
)

ADDON_SYSTEM_DIALOG_MESSAGE = PartialTemplate(
    _ADDON_SYSTEM_TEXT,
    globals={
        "request_format": _REQUEST_FORMAT_TEXT,
        "protocol_footer": _PROTOCOL_FOOTER,
    },
    template_class=DateAwareTemplate,
)

ENFORCE_JSON_FORMAT_TEMPLATE = JINJA2_ENV.from_string(_ENFORCE_JSON_FORMAT_TEXT)

MAIN_BEST_EFFORT_TEMPLATE = PartialTemplate(_MAIN_BEST_EFFORT_TEXT)

ADDON_BEST_EFFORT_TEMPLATE = PartialTemplate(_ADDON_BEST_EFFORT_TEXT)
