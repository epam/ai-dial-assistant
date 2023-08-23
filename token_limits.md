# Model Listing

The Universal API defines the following limits per deployment:

- **max_total_tokens** - The maximum number of shared tokens between the prompt and completion parts.
- **max_prompt_tokens** - The maximum number of tokens in a request.
- **max_completion_tokens** - The maximum number of tokens that the model can generate.
- **prompt_token_unit** - Defines the strategy for counting the number of tokens.
- **max_prompt_messages** - The maximum number of messages that the model can accept.
- **max_system_messages** - The maximum number of system messages that the model can accept.

Even though the limits for the assistant could be enforced, they cannot be statically defined. These limits are dynamic and depend on the following factors:

- Underlying model
- Add-ons provided in the request

## Shared context

Here are the formulas for models with shared context between prompt and completion:

- **max_total_tokens** = model total tokens - len(assistant's system message with add-ons) - add-ons' maximum dialog size<sub>per model</sub> - response overhead<sub>per model</sub> \[ - len(proxy's system message)]
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

It may be expensive to calculate limits for all combinations of assistant, model, and addons on listing request;
therefore, it appears to be more sensible to return limits only per a specified combination, e.g.
/openai/assistants/assistant-10K?model=gpt-4&add-on=wolfram.

![generic assistant](generic_assistant_context_breakdown.svg)

Internal parameters:
- The assistant's system message establishes the communication protocol between the model and other entities such as users and add-ons.
- Dialog with add-ons includes commands to invoke add-ons and responses from add-ons.
- Response overhead - are extra tokens required to differentiate between add-on invocations and replies to the user.

Which parts can be controlled by the user of the API?
- User prompt:
    - System message and last message must fit into **max_prompt_tokens**. Though, there is a request parameter that lets
models to drop the oldest messages from history if they do not fit into the provided limit and the number of dropped messages
will be returned in the response.
    - If model responses involve add-on invocations, this will be reflected in a non-empty state. This state is intended
to be opaque to the user, but it contributes to the prompt size as part of history on subsequent requests. Therefore,
the assistant should probably return the size of its message in the usage stats including state size, allowing
the user to calculate the prompt size accurately.

In a predefined assistant, the user is not able to send a system message to control the assistant's behavior:

![predefined assistant](predefined_assistant_context_breakdown.svg)

## Split context

These are formulas for models with split context (like VertexAI):

- **max_prompt_tokens** = model total prompt tokens - len(assistant's system message with add-ons) - add-ons' maximum dialog size<sub>per model</sub> \[ - len(proxy's system message)]
- **max_completion_tokens** = model total completion tokens - response overhead<sub>per model</sub>
- **prompt_token_unit** = per model unit
- **max_prompt_messages** = (I'm not sure if we need this)
- **max_system_messages** = 1 for generic assistant, 0 for predefined assistant

![generic assistant split context](generic_assistant_split_context_breakdown.svg)

## Dialog with add-ons

Dialog with add-ons consists of command and responses and may involve multiple invocations. The last command is always a reply command:
```
assistant: <list of commands>
user: <list of responses>
...
assistant: <reply command>
```
The max dialog size is an internal parameter (pair of parameters for split context models) and ensures user has guaranteed
context size.

## Tokenization/limit enforcement

There are basically 2 ways to enforce limits:
* Do tokenization and validation in the assistant.
  * Pros:
    * More accurate limits.
    * More accurate usage stats.
    * More accurate errors.
  * Cons:
    * Needs to know how to tokenize input for any model.
* Delegate tokenization and validation to a model
  * Pros
    * No need to know how to tokenize input for any model.
  * Cons
    * Need to know size of dialog with add-ons to correct max_prompt_tokens for model request<sup>1</sup> (perhaps can be retrieved from usage stats).
Otherwise, the dialog may not use all reserved tokens, that will allow more user prompt tokens to be processed.
    * Errors may be confusing if pre-calculated sizes are not up-to-date.
    * May require extra tokens as clearance for rough tokenization.

# Guarantees

- **max_total_tokens**/**max_prompt_tokens** - Lets users know how many tokens the assistant can accept to proceed with the request. At least for the last user message + system message. 
- **max_completion_tokens** - If exceeded, the response will be interrupted with a finish_reason: "length".
- **add-ons' maximum dialog size<sub>per model</sub>** - This is an internal parameter that reserves space for model's dialog with add-ons.
If exceeded, the model should explain to the user that it cannot process the request.

# Pseudocode

## Assistant logic

```python
def process_user_request():
    # user_request: dict = {
    #     "max_prompt_tokens": 100,
    #     "model": 'gpt-4',
    #     "messages": [
    #         ...
    #     ],
    #     "addons": [
    #         {
    #             "url": "https://epam-qna-application.staging.deltixhub.io/semantic-search/.well-known/ai-plugin.json"
    #         }
    #     ]
    # }

    limits = limits_service.get_limits(user_request["model"], user_request["addons"])
    # limits =
    # {
    #     "max_total_tokens": 2000,
    #     "max_addons_dialog_tokens": 500,
    #     "single_response_overhead": 5,
    #     "responses_array_overhead": 5,
    #     "reply_to_user_overhead": 5,
    # }

    user_system_message = next((m for m in user_request["messages"] if m["role"] == "system"), None)
    assistant_system_message = build_system_message(user_system_message, user_request["addons"])
    # assistant_system_message =
    # Today's date is 22-Jul-2023.
    #
    # Give answers based on facts only
    #
    # Protocol
    # The following commands are available to reply to user or find out the answer to the user's question:
    # ...

    assistant_request = {
        "model": 'gpt-4',
        "messages": [assistant_system_message] + [m for m in user_request["messages"] if m["role"] != "system"]
    }

    model_client = ModelClient(user_request["model"])
    max_user_reply_size = user_request["total_tokens"] if "total_tokens" in user_request else limits["max_total_tokens"] + limits["reply_to_user_overhead"]
    # Beforehand, we don't know what the model will do next: reply to the user or start/continue a dialogue with addons.
    max_completion_tokens = max(max_user_reply_size, limits["max_addons_dialog_tokens"])
    model_response = model_client.generate(assistant_request
                                           | {"max_prompt_tokens": user_request["max_prompt_tokens"]} if "max_prompt_tokens" in user_request else {}
                                           | {"total_tokens": max_completion_tokens})

    # Remove discarded messages from subsequent requests
    discarded_message_count = model_response["statistics"]["discarded_messages"]
    assistant_request = {
        "model": 'gpt-4',
        "messages": [assistant_system_message] + [m for m in user_request["messages"][discarded_message_count:] if m["role"] != "system"]
    }

    usage = model_response["usage"]
    commands = parse_commands(model_response["content"])
    responses = []

    prompt_tokens = usage["prompt_tokens"]
    dialog_size = usage["completion_tokens"]
    while True:
        estimated_dialog_size = dialog_size + limits["responses_array_overhead"]
        for command in commands:
            if command.name == "reply":
                return {
                    # If there are more reserved tokens left for dialog with addons than available for client reply
                    # we may end up with longer response to user than expected.
                    # There is no way to cut it correctly without a tokenizer.
                    # If model response is streamed and usage is returned for each chunk,
                    # we can interrupt the response at the end of the closest chunk.
                    "content": command.args[0],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": dialog_size,
                        "total_tokens": dialog_size + prompt_tokens
                    }
                }

            command_result = command.execute()
            # Addon session will return tokens for the result
            estimated_dialog_size += command_result.token_count
            estimated_dialog_size += limits["single_response_overhead"]

            if estimated_dialog_size > limits["max_addons_dialog_tokens"]:
                # TODO: Implement best effort logic
                # E.g. model explains that it cannot process the request
                return handle_max_addons_dialog_tokens_overflow()

            responses.append(command_result.content)

        assistant_request["messages"].append({"role": "assistant", "content": model_response["content"]})
        assistant_request["messages"].append({"role": "user", "content": {"responses": responses}})

        max_completion_tokens = max(max_user_reply_size, limits["max_addons_dialog_tokens"] - estimated_dialog_size)
        model_response = model_client.generate(assistant_request | {"total_tokens": max_completion_tokens})

        # replace estimate with actual value
        dialog_size = model_response["usage"]["total_tokens"] - prompt_tokens
```

## Client logic

Interaction with the assistant is mostly equivalent to interaction with underlying model directly, except for the following:
* Add-ons can be provided in the request that give the assistant additional capabilities.
* Limits are lower than model limits due to added overhead: system message, dialog with add-ons, additional message markup etc.
* It only supports basic chat functionality and not all functionality of the underlying model. For instance, it doesn't
support [function calls](https://openai.com/blog/function-calling-and-other-api-updates) in gpt-3.5/gpt-4.
* Usage information takes into account assistant's overhead required for use or ability to use add-ons.
* Length of generated message by assistant may contain more tokens than requested if we choose not to use tokenizers in the assistant.

```python
def send_assistant_request():
    history = [
        {
            "role": "system",
            "content": "Give answers based on facts only"
        },
        {
            "role": "user",
            "content": "What is EPAM?"
        },
        {
            "role": "assistant",
            "content": "EPAM is a leading digital transformation services and product engineering company, providing digital platform engineering and software development services to customers located around the world, primarily in North America, Europe, and Asia. They deliver business and technology transformation from start to finish, leveraging agile methodologies, customer collaboration frameworks, engineering excellence tools, hybrid teams, and their award-winning proprietary global delivery platform. They focus on building long-term partnerships with their customers in a market that is constantly challenged by the pressures of digitization through innovative strategy and scalable software solutions.",
            "custom_content": {
                "state": {
                    "invocations": [
                        {
                            "index": 0,
                            "request": "{\"commands\": [{\"command\": \"run-plugin\", \"args\": [\"epam-10k-semantic-search\", \"What is EPAM?\"]}]}",
                            "response": "{\"responses\": [{\"status\": \"SUCCESS\", \"response\": \"EPAM is a leading digital transformation services and product engineering company, providing digital platform engineering and software development services to customers located around the world, primarily in North America, Europe, and Asia. They deliver business and technology transformation from start to finish, leveraging agile methodologies, customer collaboration frameworks, engineering excellence tools, hybrid teams, and their award-winning proprietary global delivery platform. They focus on building long-term partnerships with their customers in a market that is constantly challenged by the pressures of digitization through innovative strategy and scalable software solutions.\"}]}"
                        }
                    ]
                }
            },
            "usage": {
                "prompt_tokens": 60,
                # Completion tokens include state size
                "completion_tokens": 50,
                "total_tokens": 110
            }
        },
    ]
    tokenizer = get_tokenizer(user_request["model"])
    limits = limits_service.get_limits(user_request["model"], user_request["addons"])
    # limits =
    # {
    #     "max_total_tokens": 2000
    # }

    # Leave some tokens for completion (e.g. ~40% of max_total_tokens)
    max_prompt_size = int(limits["max_total_tokens"] * 0.6)
    prompt = {
        "role": "user",
        "content": "What was EPAM's income in 2021?"
    }
    # prompt_size = system message size + last user message size
    prompt_size = tokenizer.get_token_count(history[0]["content"]) + tokenizer.get_token_count(prompt["content"])

    if prompt_size > max_prompt_size:
        raise Exception(f'Prompt is too long. Max tokens: {max_prompt_size}, actual: {prompt_size}')

    previous_assistant_response = history[-1]
    history_size = previous_assistant_response["usage"]["total_tokens"]

    if history_size + prompt_size > limits["max_total_tokens"]:
        print("Warning: The history doesn't fit into the model and as a result, some old messages will be ignored.")

    user_request: dict = {
        "max_prompt_tokens": max_prompt_size,
        "model": 'gpt-4',
        "messages": history + [prompt],
        "addons": [
            {
                "url": "https://epam-qna-application.staging.deltixhub.io/semantic-search/.well-known/ai-plugin.json"
            }
        ]
    }

    assistant_response = assistant_service.generate(user_request)["choices"][0]

    finish_reason = assistant_response["finish_reason"]
    print(f'Got response from assistant with finish_reason {finish_reason}: {assistant_response["message"]["content"]}')
```

### Notes

Add-on invocations may not always finish successfully due to various reasons, such as: network issues, insufficient
context size to process user query, etc. Therefore, the assistant is expected to provide user answer based on available
information. If there is not enough available tokens for the answer, response will contain finish_reason: "length".
Continuation of aborted token sequence is not currently supported.  