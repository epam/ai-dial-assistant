from typing_extensions import override

from aidial_assistant.json_stream.exceptions import invalid_sequence_error
from aidial_assistant.json_stream.json_node import AtomicNode

TRUE_STRING = "true"
FALSE_STRING = "false"
TYPE_STRING = "boolean"


class JsonBoolean(AtomicNode[bool]):
    def __init__(self, raw_data: str, pos: int):
        super().__init__(raw_data, pos)
        self._value: bool = JsonBoolean._parse_boolean(raw_data, pos)

    @override
    def type(self) -> str:
        return TYPE_STRING

    @override
    def value(self) -> bool:
        return self._value

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == "t" or char == "f"

    @staticmethod
    def _parse_boolean(string: str, char_position: int) -> bool:
        if string == TRUE_STRING:
            return True

        if string == FALSE_STRING:
            return False

        raise invalid_sequence_error(TYPE_STRING, string, char_position)
