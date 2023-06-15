import traceback

from colorama import Back, Fore, Style
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


def print_red(msg: str, **args):
    print(Fore.RED + msg + Style.RESET_ALL, flush=True, **args)


def print_error(msg: str, **args):
    print(Fore.RED + msg + Style.RESET_ALL, flush=True, **args)


def print_warning(msg: str, **args):
    print(Fore.YELLOW + msg + Style.RESET_ALL, flush=True, **args)


def print_info(msg: str, **args):
    print(Fore.YELLOW + msg + Style.RESET_ALL, flush=True, **args)


def print_system(msg: str, **args):
    print(Back.LIGHTBLACK_EX + msg + Style.RESET_ALL, flush=True, **args)


def print_human(msg: str, **args):
    print(Fore.GREEN + msg + Style.RESET_ALL, flush=True, **args)


def print_ai(msg: str, **args):
    print(Fore.BLUE + msg + Style.RESET_ALL, flush=True, **args)


def print_exception():
    print(Fore.RED, flush=True)
    traceback.print_exc()
    print(Style.RESET_ALL, flush=True)


def print_base_message(msg: BaseMessage, **args):
    if isinstance(msg, HumanMessage):
        print_human(msg.content, **args)
    elif isinstance(msg, AIMessage):
        print_ai(msg.content, **args)
    elif isinstance(msg, SystemMessage):
        print_system(msg.content, **args)
    else:
        print(msg.content, **args)
