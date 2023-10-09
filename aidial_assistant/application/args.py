import argparse
from pathlib import Path
from typing import NamedTuple

from aidial_assistant.application.project_conf import (
    ChatConf,
    OpenAIConf,
    read_conf,
)


class Args(NamedTuple):
    chat_conf: ChatConf
    openai_conf: OpenAIConf


def add_yaml_conf(
    parser: argparse.ArgumentParser, name: str, default: str, help: str
):
    parser.add_argument(
        name,
        type=str,
        metavar="YAML",
        default=default,
        help=help + f" (default: {default})",
    )


def parse_args(config_dir: Path) -> Args:
    parser = argparse.ArgumentParser()

    add_yaml_conf(
        parser,
        "--chat-conf",
        default=str(config_dir / "chat.yaml"),
        help="Path to chat configuration file",
    )
    add_yaml_conf(
        parser,
        "--openai-conf",
        default=str(config_dir / "open_ai.yaml"),
        help="Path to OpenIA configuration file",
    )

    parsed_args, _ = parser.parse_known_args()

    chat_conf = read_conf(ChatConf, Path(parsed_args.chat_conf))
    openai_conf = read_conf(OpenAIConf, Path(parsed_args.openai_conf))

    args = Args(
        chat_conf=chat_conf,
        openai_conf=openai_conf,
    )

    return args
