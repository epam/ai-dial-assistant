## Overview

The DIAL Assistant Service is designed to respond to user queries, like ChatGPT. It is accessible via [DIAL API](https://epam-rail.com/dial_api).
The service’s distinctive feature is its ability to utilize addons provided in the user request, enhancing its
capability to gather and process information. Upon receiving a user request, the service employs the specified LLM to
interpret and respond to the inquiry. Along with user request it instructs the model on how to apply the provided addons
to garner additional information. If the model decides to use an addon to seek specific details, the Assistant Service
promptly executes this task and channels the acquired data back to the model. This iterative procedure continues, with
the model leveraging the addons to assemble more information until a thorough and informed response to the user’s query
is generated. In essence, the DIAL Assistant Service is a versatile tool that combines the power of a given model with
the extended capabilities of various addons to deliver comprehensive and accurate answers.

## Usage example

```python
import os
import typing

import openai

if __name__ == "__main__":
    response: typing.Iterable[typing.Any] = openai.ChatCompletion.create(
        api_base="https://<assistant-service-host>",
        api_type="azure",
        api_version="2023-03-15-preview",
        api_key=os.environ["RAIL_PROXY_API_KEY"],
        engine="assistant",
        model="gpt-4",
        temperature=0,
        timeout=300,
        messages=[
            {"role": "user", "content": "What's up?"},
        ],
        addons=[
            {
                "url": "https://<addon-host>/.well-known/ai-plugin.json"
            }
        ],
    )

    print(response)
```


## Developer environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=1.6.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

## Build

To install the package dependencies and create a virtual environment run:

```sh
make install
```

## Run

Run the development server:

```sh
make serve
```

### Make on Windows
As of now, Windows distributions do not include the make tool. To run make commands, the tool can be installed using
the following command (since [Windows 10](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):
```sh
winget install GnuWin32.Make
```
For convenience, the tool folder can be added to the PATH environment variable as `C:\Program Files (x86)\GnuWin32\bin`.
The command definitions inside Makefile should be cross-platform to keep the development environment setup simple.

## Environment Variables

Copy .env.example to .env and customize it for your environment:

| Variable         | Default                  | Description                                            |
|------------------|--------------------------|--------------------------------------------------------|
| LOG_LEVEL        | INFO                     | Log level. Use DEBUG for dev purposes and INFO in prod |
| CONFIG_DIR       | aidial_assistant/configs | Configuration directory                                |
| OPENAPI_API_BASE | N/A                      | OpenAI API Base                                        |

### Docker

Run the server in Docker:

```sh
make docker_serve
```

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally:

```sh
make test
```

## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```
