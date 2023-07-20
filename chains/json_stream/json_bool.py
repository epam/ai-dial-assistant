import json

from typing_extensions import override

from chains.json_stream.json_node import PrimitiveNode

TRUE_STRING = "true"
FALSE_STRING = "false"


class JsonBoolean(PrimitiveNode[bool]):
    def __init__(self, raw_data: str, char_position: int):
        super().__init__(char_position)
        self._raw_data = raw_data
        self._value: bool = json.loads(raw_data)

    @override
    def type(self) -> str:
        return 'boolean'

    @override
    def raw_data(self) -> str:
        return self._raw_data

    @override
    def value(self) -> bool:
        return self._value

    @staticmethod
    def is_bool(char: str) -> bool:
        return char == 't' or char == 'f'
