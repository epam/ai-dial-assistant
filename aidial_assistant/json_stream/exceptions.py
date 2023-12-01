class JsonParsingException(Exception):
    def __init__(self, message: str, char_position: int):
        super().__init__(
            f"Failed to parse json string at position {char_position}: {message}"
        )


def unexpected_symbol_error(
    char: str, char_position: int
) -> JsonParsingException:
    return JsonParsingException(f"Unexpected symbol {char}.", char_position)


def unexpected_end_of_stream_error(char_position: int) -> JsonParsingException:
    return JsonParsingException("Unexpected end of stream.", char_position)


def invalid_sequence_error(
    sequence: str, char_position: int
) -> JsonParsingException:
    return JsonParsingException(f"Invalid sequence {sequence}.", char_position)
