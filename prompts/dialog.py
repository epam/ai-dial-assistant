from langchain import PromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


# Compute today's date in the format "DD Mmm YYYY"
def get_today_date():
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d %b %Y")


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
""".strip()

system_template = """
Your training data is up-to-date until September 2021.
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
 - NAME must be one of the following plugins:
{%- for name, description in tools.items() %}
    * {{name}}: {{description | decap}}
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
Your training data is up-to-date until September 2021.
Today's date is {{today_date}}.

{%- if system_prefix %}
{{system_prefix}}
{%- endif %}

Protocol
The following commands are available to reply to user or find out the answer to the user's question:
{%- for name, command in commands.items() %}
> {{name}}
{%- if command.description | decap %}
{{command.description}}{% endif %}
{%- if command.result %}
The command returns {{command.result | decap}}
{%- endif %}
{%- if command.args %}
Arguments:
{%- for arg in command.args %}
 - {{ arg }}
{%- endfor %}
{%- endif %}
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

MAIN_SYSTEM_DIALOG_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=system_template,
        template_format="jinja2",
        input_variables=["system_prefix", "tools"],
        partial_variables={
            "today_date": get_today_date(),
            "request_response": request_response
        },
    )
)

PLUGIN_SYSTEM_DIALOG_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=plugin_system_template,
        template_format="jinja2",
        input_variables=["system_prefix", "commands"],
        partial_variables={
            "today_date": get_today_date(),
            "request_response": request_response
        },
    )
)

RESP_DIALOG_PROMPT = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=resp_template,
        template_format="jinja2",
        input_variables=["response"]
    )
)
