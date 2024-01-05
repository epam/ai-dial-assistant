from aidial_assistant.application.prompts import (
    ADDON_BEST_EFFORT_TEMPLATE,
    MAIN_BEST_EFFORT_TEMPLATE,
)


def test_main_best_effort_prompt():
    actual = MAIN_BEST_EFFORT_TEMPLATE.build(
        addons={"addon name": "Addon description"}
    ).render(
        error="<error>",
        message="<message>",
        dialogue=[{"role": "<role>", "content": "<content>"}],
    )

    assert (
        actual
        == """You were allowed to use the following addons to answer the query below.

=== ADDONS ===

* addon name - addon description

=== QUERY ===

<message>

=== ADDONS REQUESTS AND RESPONSES ===

<role>: <content>

However, the follow-up requests failed with the following error:
> <error>

Please respond to the query using the available information, and explaining that the use of the addons was not possible due to the error."""
    )


def test_main_best_effort_prompt_with_empty_dialogue():
    actual = MAIN_BEST_EFFORT_TEMPLATE.build(
        addons={"addon name": "Addon description"}
    ).render(
        error="<error>",
        message="<message>",
        dialogue=[],
    )

    assert (
        actual
        == """You were allowed to use the following addons to answer the query below.

=== ADDONS ===

* addon name - addon description

=== QUERY ===

<message>

However, the follow-up requests failed with the following error:
> <error>

Please respond to the query using the available information, and explaining that the use of the addons was not possible due to the error."""
    )


def test_plugin_best_effort_prompt():
    actual = ADDON_BEST_EFFORT_TEMPLATE.build(api_schema="<api schema>").render(
        error="<error>",
        message="<message>",
        dialogue=[{"role": "<role>", "content": "<content>"}],
    )

    assert (
        actual
        == """You were allowed to use the following API to answer the query below.

=== API ===

```typescript
<api schema>
```

=== QUERY ===

<message>

=== API REQUESTS AND RESPONSES ===

<role>: <content>

However, the follow-up requests failed with the following error:
> <error>

Please respond to the query using the available information, and explaining that the use of the API was not possible due to the error."""
    )


def test_plugin_best_effort_prompt_with_empty_dialogue():
    actual = ADDON_BEST_EFFORT_TEMPLATE.build(api_schema="<api schema>").render(
        error="<error>",
        message="<message>",
        dialogue=[],
    )

    assert (
        actual
        == """You were allowed to use the following API to answer the query below.

=== API ===

```typescript
<api schema>
```

=== QUERY ===

<message>

However, the follow-up requests failed with the following error:
> <error>

Please respond to the query using the available information, and explaining that the use of the API was not possible due to the error."""
    )
