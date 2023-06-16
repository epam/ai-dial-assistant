def indent(text, num_spaces, start_symbol=None):
    indentation = " " * num_spaces
    if start_symbol is not None:
        indentation = indentation + start_symbol + " "
    lines = text.splitlines()
    indented_lines = [indentation + line for line in lines]
    return "\n".join(indented_lines)


def decapitalize(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]
