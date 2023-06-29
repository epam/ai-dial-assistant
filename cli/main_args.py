import argparse
from pathlib import Path
from typing import NamedTuple

from conf.project_conf import ChatConf, OpenAIConf, read_conf


class Args(NamedTuple):
    chat_conf: ChatConf
    openai_conf: OpenAIConf


def add_yaml_conf(parser: argparse.ArgumentParser, name: str, default: str, help: str):
    parser.add_argument(
        name,
        type=str,
        metavar="YAML",
        default=default,
        help=help + f" (default: {default})",
    )


def parse_args(root_path: str = ".") -> Args:
    parser = argparse.ArgumentParser()

    add_yaml_conf(
        parser,
        "--chat-conf",
        default=f"{root_path}/configs/chat.yaml",
        help="Path to chat configuration file",
    )
    add_yaml_conf(
        parser,
        "--openai-conf",
        default=f"{root_path}/configs/open_ai.yaml",
        help=f"Path to OpenIA configuration file",
    )

    parsed_args, argv = parser.parse_known_args()

    chat_conf = read_conf(ChatConf, Path(parsed_args.chat_conf))
    openai_conf = read_conf(OpenAIConf, Path(parsed_args.openai_conf))

    args = Args(
        chat_conf=chat_conf,
        openai_conf=openai_conf,
    )

    return args
