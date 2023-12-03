from typing_extensions import override

from aidial_assistant.json_stream.exceptions import invalid_sequence_error
from aidial_assistant.json_stream.json_node import AtomicNode

NULL_STRING = "null"


class JsonNull(AtomicNode[None]):
    def __init__(self, raw_data: str, pos: int):
        super().__init__(raw_data, pos)
        if raw_data != NULL_STRING:
            raise invalid_sequence_error(NULL_STRING, raw_data, pos)

    @override
    def type(self) -> str:
        return NULL_STRING

    @override
    def value(self) -> None:
        return None

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == "n"
