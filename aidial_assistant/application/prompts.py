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


request_response = """
ALWAYS reply with a JSON containing an array of available commands. You must not use natural language:
{
  "commands": [
    {
      "command": "<command name>",
      "args": [
        <array of arguments>
      ]
    }
  ]
}

Example:
{"commands": [{"command": "reply", "args": ["Hello, world!"]}]}
""".strip()

system_template = """
Today's date is {{today_date}}.

{%- if system_prefix %}
{{system_prefix}}
{%- endif %}

Protocol
The following commands are available to reply to user or find out the answer to the user's question:
{%- if tools %}
> run-plugin
The command runs a specified plugin to solve a one-shot task written in natural language.
Plugins do not see current conversation and require all details to be provided in the query to solve the task.
The command returns the result of the plugin call.
Arguments:
 - NAME is one of the following plugins:
{%- for name, description in tools.items() %}
    * {{name}} - {{description | decap}}
{%- endfor %}
 - QUERY is a string formulating the query to the plugin.
{%- endif %}
> reply
The command delivers ultimate result to the user.
Arguments:
 - MESSAGE is a string containing response for user.

{{request_response}}
""".strip()

plugin_system_template = """
Today's date is {{today_date}}.

Service
API_DESCRIPTION:
{{api_description}}

API_SCHEMA:
```typescript
{{api_schema}}}
```

Protocol
The following commands are available to reply to user or find out the answer to the user's question:
{%- for command_name in command_names %}
> {{command_name}}
Arguments:
 - <JSON dict according to the API_SCHEMA>
{%- endfor %}
> reply
The command delivers ultimate result to the user
Arguments:
 - MESSAGE is a string containing response for user.

{{request_response}}
""".strip()

resp_template = """
{{response}}
**Remember to reply with a JSON with commands**
""".strip()

open_api_plugin_template = """
Service
API_DESCRIPTION:
{{api_description}}

API_SCHEMA:
```typescript
{{api_schema}}}
```
""".strip()

MAIN_SYSTEM_DIALOG_MESSAGE = JINJA2_ENV.from_string(
    system_template,
    globals={
        "request_response": request_response,
    },
    template_class=DateAwareTemplate,
)

PLUGIN_SYSTEM_DIALOG_MESSAGE = JINJA2_ENV.from_string(
    plugin_system_template,
    globals={
        "request_response": request_response,
    },
    template_class=DateAwareTemplate,
)

RESP_DIALOG_PROMPT = JINJA2_ENV.from_string(resp_template)
