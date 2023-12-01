import json

from typing_extensions import override

from aidial_assistant.json_stream.exceptions import invalid_sequence_error
from aidial_assistant.json_stream.json_node import AtomicNode


class JsonNumber(AtomicNode[float | int]):
    def __init__(self, raw_data: str, char_position: int):
        super().__init__(raw_data, char_position)
        self._value: float | int = JsonNumber._parse_number(
            raw_data, char_position
        )

    @override
    def type(self) -> str:
        return "number"

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
            raise invalid_sequence_error(string, char_position)
