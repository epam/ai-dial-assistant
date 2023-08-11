# Model Listing

The Universal API defines the following limits per deployment:

- **max_total_tokens** - The maximum number of shared tokens between the prompt and completion parts.
- **max_prompt_tokens** - The maximum number of tokens in a request.
- **max_completion_tokens** - The maximum number of tokens that the model can generate.
- **prompt_token_unit** - Defines the strategy for counting the number of tokens.
- **max_prompt_messages** - The maximum number of messages that the model can accept.
- **max_system_messages** - The maximum number of system messages that the model can accept.

Even though the limits for the assistant can be enforced, they cannot be statically defined. These limits are dynamic and depend on the following factors:

- Underlying model
- Add-ons provided in the request
- max_completion_tokens if the user can reduce it

If the /openai/models could accept the completion request, then all required limits would be adjusted accordingly. Here are the formulas:

- **max_total_tokens** = model total tokens - len(assistant's system message with add-ons) - add-ons' maximum dialog size<sub>per model</sub> \[ - len(proxy's system message)]
- **max_prompt_tokens** = max_total_tokens - max_completion_tokens
- **max_completion_tokens** = per model limit (user can make it smaller to get more for prompt)
- **prompt_token_unit** = per model unit
- **max_prompt_messages** = (I'm not sure if we need this)
- **max_system_messages** = 1 for generic assistant, 0 for predefined assistant

Currently, the proxy doesn't know which system message is used by the assistant or the overhead added by add-on descriptions.
Therefore, to calculate the length in tokens of the assistant's system message with add-ons, the proxy needs the following information:

- System message size per model or, alternatively, the system message itself.
- Add-on description overhead (e.g., each add-on listing includes indentation, separators, etc.).
- Parse add-on JSON to extract required fields: name and description.

For models like gpt-3.5/gpt-4, to calculate tokens precisely, the entire text has to be tokenized.
That is, len(text<sub>1</sub> + text<sub>2</sub>) may not be equal to len(text<sub>1</sub>) + len(text<sub>2</sub>).
Therefore, I'm more inclined to delegate the calculation of the overhead added by the assistant's system message with
add-ons to the assistant itself by adding a new endpoint to the assistant API.

# Guarantees

- **max_prompt_tokens** - Lets users know how many tokens the assistant can accept to proceed with the request. At least for the last user message + system message. 
- **max_completion_tokens** - This is a reserved space for the assistant's response. If exceeded, the response will be interrupted with a finish_reason: "length".
- **add-ons' maximum dialog size<sub>per model</sub>** - This is an internal parameter that reserves space for model dialog with add-ons.
If exceeded, the model should explain to the user that it cannot process the request.
