from typing import AsyncIterator

from typing_extensions import override

from chains.json_stream.json_node import PrimitiveNode, unexpected_symbol_error

NULL_STRING = "null"


class JsonNull(PrimitiveNode[None]):
    def __init__(self, raw_data: str, char_position: int):
        super().__init__(char_position)
        if raw_data != 'null':
            raise unexpected_symbol_error(raw_data, char_position)

    @override
    def type(self) -> str:
        return 'null'

    @override
    def raw_data(self) -> str:
        return NULL_STRING

    @override
    def value(self) -> None:
        return None

    @staticmethod
    def is_null(char: str) -> bool:
        return char == 'n'
