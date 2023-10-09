PORT ?= 5000
IMAGE_NAME ?= ai-dial-assistant
PLATFORM ?= linux/amd64
ARGS=

.PHONY: all install build serve docker_serve clean lint format test

all: build

install:
	poetry install

build: install
	poetry build

serve: install
	poetry run uvicorn "aidial_assistant.app:app" --reload --host "0.0.0.0" --port $(PORT) --env-file ./.env

docker_serve:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME):latest .
	docker run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):latest

clean:
	poetry run clean
	poetry env remove --all

lint: install
	poetry run nox -s lint

format: install
	poetry run nox -s format

# Run `make test ARGS="-v --durations=0 -rA"` to see stderr/stdout of each test.
test: install
	poetry run nox -s test -- $(ARGS)

help:
	@echo "===================="
	@echo "build                        - build the source and wheels archives"
	@echo "clean                        - clean virtual env and build artifacts"
	@echo "-- LINTING --"
	@echo "format                       - run code formatters"
	@echo "lint                         - run linters"
	@echo "-- RUN --"
	@echo "serve                        - run the dev server locally"
	@echo "docker_serve                 - run the dev server from the docker"
	@echo "-- TESTS --"
	@echo "test                         - run unit tests"
