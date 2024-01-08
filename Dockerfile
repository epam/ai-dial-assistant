# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-alpine AS builder

ARG POETRY_VERSION=1.6.1

# Vulnerability fix: CVE-2023-5363
RUN apk update && apk upgrade --no-cache libcrypto3 libssl3

# Install alpine-sdk to compile some langchain dependencies (numexpr, numpy)
RUN apk add --no-cache alpine-sdk linux-headers

RUN pip install --upgrade pip
RUN pip install poetry==$POETRY_VERSION

# Test if Poetry is installed in the expected path
RUN echo "Poetry version:" && poetry --version

# Set the working directory for the app
WORKDIR /app
COPY pyproject.toml poetry.lock poetry.toml ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --only main

COPY aidial_assistant/ ./aidial_assistant/

# Use a multi-stage build to run the server
FROM python:3.11-alpine

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV LOG_LEVEL=INFO

# Vulnerability fix: CVE-2023-5363
RUN apk update && apk upgrade --no-cache libcrypto3 libssl3

# Install libstdc++ to run the compiled dependencies
RUN apk add --no-cache libstdc++

# Create a non-root user with an explicit UID
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1001 --disabled-password --gecos "" appuser
# Copy and add permission to access the /app folder
COPY --chown=appuser --from=builder /app /app
WORKDIR /app

# "Activate" the virtual environment
ENV PATH=/app/.venv/bin:$PATH

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
EXPOSE 5000

USER appuser
CMD ["uvicorn", "aidial_assistant.app:app", "--host", "0.0.0.0", "--port", "5000"]
