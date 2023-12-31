import json

from typing_extensions import override

from aidial_assistant.json_stream.exceptions import invalid_sequence_error
from aidial_assistant.json_stream.json_node import AtomicNode

TYPE_STRING = "number"


class JsonNumber(AtomicNode[float | int]):
    def __init__(self, raw_data: str, pos: int):
        super().__init__(raw_data, pos)
        self._value: float | int = JsonNumber._parse_number(raw_data, pos)

    @override
    def type(self) -> str:
        return TYPE_STRING

    @override
    def value(self) -> float | int:
        return self._value

    @staticmethod
    def starts_with(char: str) -> bool:
        return char.isdigit() or char == "-"

    @staticmethod
    def _parse_number(string: str, char_position: int) -> float | int:
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            raise invalid_sequence_error(TYPE_STRING, string, char_position)
