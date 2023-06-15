import re
from typing import Callable, List, NamedTuple, TypeVar


def format_safe(format_string: str, repl: dict) -> str:
    try:
        output_string = format_string.format(**repl)
        return output_string
    except KeyError as e:
        keys = ", ".join(list(repl.keys()))
        print(format_string)
        raise ValueError(
            f"Unfilled hole in a format string: {str(e)}. Expected one of: {keys}"
        )


def indent(text, num_spaces, start_symbol=None):
    indentation = " " * num_spaces
    if start_symbol is not None:
        indentation = indentation + start_symbol + " "
    lines = text.splitlines()
    indented_lines = [indentation + line for line in lines]
    return "\n".join(indented_lines)


def replace_prefix(string: str, old_prefix: str, new_prefix: str) -> str:
    if string.startswith(old_prefix):
        return new_prefix + string.removeprefix(old_prefix)

    raise Exception(f"The string '{string}' is expected to have prefix '{old_prefix}'")


T = TypeVar("T")


def split_into_chunks(lst: List[T], n: int) -> List[List[T]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def split_text_into_chunks_by_lines(content: str, max_lines: int) -> List[str]:
    delim = "\n"
    return list(
        map(
            lambda xs: delim.join(xs),
            split_into_chunks(content.split(delim), max_lines),
        )
    )


class TextChunk(NamedTuple):
    content: str
    token_count: int
    lines_count: int


def split_text_into_chunks_by_tokens(
    content: str, token_counter: Callable[[str], int], max_tokens: int
) -> List[TextChunk]:
    lines = content.splitlines()

    if len(lines) == 0:
        return []

    ret: List[TextChunk] = []

    acc = lines[0]
    lines_count = 1

    for line in lines[1:]:
        new_str = acc + "\n" + line
        if token_counter(new_str) > max_tokens:
            ret.append(
                TextChunk(
                    content=acc, token_count=token_counter(acc), lines_count=lines_count
                )
            )
            acc = line
            lines_count = 1
        else:
            acc = new_str
            lines_count += 1

    ret.append(
        TextChunk(content=acc, token_count=token_counter(acc), lines_count=lines_count)
    )

    return ret


def extract_code_block(s: str) -> str:
    s1 = re.sub(r"^```[\w-]*\n", "", s.strip())
    s2 = re.sub(r"```\s*$", "", s1)
    return s2.strip()
