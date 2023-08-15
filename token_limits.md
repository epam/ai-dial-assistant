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
- **max_completion_tokens**

If the /openai/models endpoint could accept the above information, all required limits will be adjusted accordingly.
Here are the formulas:

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
- Model response markup (e.g., a reply command containing the response text).
- Parse add-on JSON to extract required fields: name and description.

For models such as gpt-3.5/gpt-4, the entire text must be tokenized to calculate tokens precisely.
In other words, len(text<sub>1</sub> + text<sub>2</sub>) may not be equal to len(text<sub>1</sub>) + len(text<sub>2</sub>).
Therefore, I'm more inclined to delegate the calculation of the overhead added by the assistant's system message with
add-ons to the assistant itself by adding a new endpoint to the assistant API.

![generic assistant](generic_assistant_context_breakdown.svg)

Internal parameters:
- The assistant's system message establishes the communication protocol between the model and other entities such as users and add-ons..
- Dialog with add-ons includes commands to invoke add-ons and responses from add-ons.
- Response overhead - are extra tokens required to differentiate between add-on invocations and replies to the user.

Which parts can be controlled by the user of the API?
- User prompt:
    - System message and last message must fit into **max_prompt_tokens**, the rest of the history can be dropped and
the result will contain the number of dropped messages.
    - If model responses involve add-on invocations, this will be reflected in a non-empty state. This state is intended
to be opaque to the user, but it contributes to the prompt size. Therefore, the assistant should return the size of its
message in the usage stats, allowing the user to calculate the prompt size accurately.
- Final model response size - **max_completion_tokens** - can be reduced by the user to get more tokens for the prompt.

In a predefined assistant user is not able to send a system message that controls the assistant's behavior:

![predefined assistant](predefined_assistant_context_breakdown.svg)

# Guarantees

- **max_prompt_tokens** - Lets users know how many tokens the assistant can accept to proceed with the request. At least for the last user message + system message. 
- **max_completion_tokens** - This is a reserved space for the assistant's response. If exceeded, the response will be interrupted with a finish_reason: "length".
- **add-ons' maximum dialog size<sub>per model</sub>** - This is an internal parameter that reserves space for model's dialog with add-ons.
If exceeded, the model should explain to the user that it cannot process the request.
