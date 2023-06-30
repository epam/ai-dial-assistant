import json

from typing_extensions import override

from chains.json_stream.json_node import PrimitiveNode


class JsonNumber(PrimitiveNode[float | int]):
    def __init__(self, raw_data: str, char_position: int):
        super().__init__(char_position)
        self._raw_data = raw_data
        self._value: float | int = json.loads(raw_data)

    @override
    def type(self) -> str:
        return 'number'

    @override
    def raw_data(self) -> str:
        return self._raw_data

    @override
    def value(self) -> float | int:
        return self._value

    @staticmethod
    def is_number(char: str) -> bool:
        return char.isdigit() or char == '-'
