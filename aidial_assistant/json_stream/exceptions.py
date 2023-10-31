class JsonParsingException(Exception):
    pass


def unexpected_symbol_error(
    char: str, char_position: int
) -> JsonParsingException:
    return JsonParsingException(
        f"Failed to parse json string: unexpected symbol {char} at position {char_position}"
    )
