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
You must always reply with a list of commands to execute:
{ "commands": [COMMAND_1, COMMAND_2, ...] }

The command responses are returned in the following format (single response per each command):
{ "responses": [{
    "id": RESPONSE_ID,
    "status": SUCCESS|ERROR,
    "response": RESPONSE
}]}
or if contract is violated:
{ "error": ERROR_MESSAGE }
""".strip()

system_template = """
Your training data goes up until September 2021.
Today is {{today_date}}.

{%- if system_prefix %}

{{system_prefix}}{% endif %}

The following list of commands is available to you to answer the user's questions:
{%- if tools %}
* {"command": "run-plugin", "args": [NAME, QUERY]}
The command runs a specified plugin to solve a one-shot task written in natural language.
Plugins do not see current conversation and require all details to be provided in the query to solve the task.
The command returns result of the plugin call.
QUERY is a string formulating the query to the plugin.
NAME must be one of the following plugins:
{%- for name, description in tools.items() %}
    - {{name}}: {{description | decap}}
{%- endfor %}
{%- endif %}
{%- for name, command in commands.items() %}
* {"command": "{{name}}", "args": [{{ command.args | join(", ") }}]}
{%- if command.description %}
{{command.description}}{% endif %}
{%- if command.result %}
The command returns {{command.result | decap}}{% endif %}
{%- endfor %}
* {"command": "say-or-ask", "args": [MESSAGE_OR_QUESTION]}
The command sends a message to the user, e.g., to ask a question or a clarification or to provide a result.
The commands returns the user's response.

{{request_response}}

The command say-or-ask is the only way to communicate with the user.

Your primary goal is to answer the user's questions.
1. You must answer the user's questions in a thorough and comprehensive manner.
2. If you don't understand the question, you must ask the user for clarification.

Start the dialogue by saying "How can I help you?".

{{reinforce_format}}
""".strip()

plugin_system_template = """
Act as a helpful assistant. Your training data goes up until September 2021.
Today is {{today_date}}.

{%- if system_prefix %}
{{system_prefix}}{% endif %}

The following list of commands is available to you to answer the user's questions:
* {"command": "end-dialog", "args": [STRING_RESULT_OR_EXPLANATION]}
The command stops the dialogue when the result is ready, or explains why the user's request cannot be processed.
{%- for name, command in commands.items() %}
* {"command": "{{name}}", "args": [{{ command.args | join(", ") }}]}
{%- if command.description | decap %}
{{command.description}}{% endif %}
{%- if command.result %}
The command returns {{command.result | decap}}{% endif %}
{%- endfor %}

{{request_response}}

The goal is to find out the answer, provided that you are given all the required input.
No command arguments can be guessed. If any is unknown, you must invoke the end-dialog command requesting clarification.

The user question:

> {{query}}

{{reinforce_format}}
""".strip()

resp_template = """
{{responses}}
{{reinforce_format}}
""".strip()

reinforce_format = "Important: you must reply with a valid JSON containing a list of commands. You must not use natural language."

open_ai_plugin_template = """
Solve the task using the service defined below.

Description of the service:

{{description_for_model}}

Service base URL: {{url}}

Open API specification:

{{open_api}}
""".strip()

open_api_plugin_template = """
You are given API schema and description of the service.
You must use the service to follow the user's instructions.

API_DESCRIPTION:
{{api_description}}

API_SCHEMA:

```typescript
{{api_schema}}}
```
""".strip()

SYSTEM_DIALOG_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=system_template,
        template_format="jinja2",
        input_variables=["system_prefix", "tools", "commands"],
        partial_variables={
            "today_date": get_today_date(),
            "request_response": request_response,
            "reinforce_format": reinforce_format,
        },
    )
)

PLUGIN_SYSTEM_DIALOG_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=plugin_system_template,
        template_format="jinja2",
        input_variables=["system_prefix", "commands", "query"],
        partial_variables={
            "today_date": get_today_date(),
            "request_response": request_response,
            "reinforce_format": reinforce_format,
        },
    )
)

RESP_DIALOG_PROMPT = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=resp_template,
        template_format="jinja2",
        input_variables=["responses"],
        partial_variables={
            "reinforce_format": reinforce_format,
        },
    )
)
