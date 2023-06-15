from langchain import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
)

open_api_selector_template = """
You are a helpful AI Assistant.
Please provide a command and args from the given API based on the user's instructions.

API_DESCRIPTION:
{{api_description}}

API_SCHEMA: ```typescript
{{api_schema}}}
```

USER_INSTRUCTIONS: {{query}}

Reply with the JSON object following the format:

```json
{"command": <command_name>, "args": <command_args>}
```

The <command_args> must confirm to the input argument schema of the corresponding command declared in API_SCHEMA.

All string arguments must be wrapped in double quotes and properly escaped.
You MUST strictly comply to the types indicated by the provided schema, including all required args.

If you don't have sufficient information to call the function due to things like requiring specific uuid's, you can reply with the following JSON:

```json
{"user_question": "Concise response requesting the additional information that would make calling the function successful"}
```

Begin
-----
""".strip()

OPEN_API_SELECTOR_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=open_api_selector_template,
        template_format="jinja2",
        input_variables=["api_schema", "api_description", "query"],
    )
)

open_api_summary_template = """
You are a helpful AI assistant trained to answer user queries from API responses.
You attempted to call an API, which resulted in:
API_RESPONSE: {{api_response}}

USER_COMMENT: {{query}}

If the API_RESPONSE can answer the USER_COMMENT respond with the following JSON object:

```json
{"success": "Human-understandable synthesis of the API_RESPONSE"}
```

Otherwise respond with the following JSON object:

```json
{"error": "What you did and a concise statement of the resulting error. If it can be easily fixed, provide a suggestion"}
```

The person you are responding to CANNOT see the API_RESPONSE, so if there is any relevant information there you must include it in your response.

Begin:
---
""".strip()

OPEN_API_SUMMARY_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=open_api_summary_template,
        template_format="jinja2",
        input_variables=["api_response", "query"],
    )
)
