from abc import ABC, abstractmethod
from queue import Queue
from typing import Iterator, Any, Dict, TypeVar, Generic

from typing_extensions import override


T = TypeVar("T")


class PeekableIterator(ABC, Iterator[T], Generic[T]):
    @abstractmethod
    def peek(self) -> T:
        pass


class Tokenator(PeekableIterator[str]):
    def __init__(self, source: Iterator[str]):
        self._source = source
        self._token: str = ""
        self._offset: int = 0
        self._position: int = 0

    @override
    def __next__(self) -> str:
        result = self.peek()
        self._offset += 1
        return result

    @override
    def peek(self) -> str:
        while self.is_last_in_token():
            self._token = next(self._source)
            self._offset = 0
            self._position += len(self._token)
        return self._token[self._offset]

    def is_last_in_token(self) -> bool:
        return self._offset == len(self._token)

    def position(self) -> int:
        return self._position + self._offset


class JsonParser(ABC):
    @abstractmethod
    def parse(self, stream: Tokenator):
        pass


class JsonString(JsonParser, Iterator[str]):
    def __init__(self):
        self.listener: Queue = Queue()

    @staticmethod
    def token() -> str:
        return '"'

    @override
    def __next__(self) -> str:
        result = self.listener.get()
        if result is None:
            raise StopIteration
        return result

    @override
    def parse(self, stream: Tokenator):
        for char in JsonString.read(stream):
            self.listener.put(char)
        self.listener.put(None)

    @staticmethod
    def read(stream: Tokenator) -> Iterator[str]:
        char = next(stream)
        if not char == JsonString.token():
            raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
        result = ""
        while True:
            char = next(stream)
            if char == JsonString.token():
                break

            result += JsonString._escape(stream) if char == '\\' else char
            if stream.is_last_in_token():
                yield result
                result = ""

        if result:
            yield result

    @staticmethod
    def _escape(stream: Tokenator) -> str:
        char = next(stream)
        if char == 'u':
            unicode_sequence = ''.join(next(stream) for _ in range(4))
            return str(int(unicode_sequence, 16))
        if char in '"\\/':
            return char
        if char == 'b':
            return '\b'
        elif char == 'f':
            return '\f'
        elif char == 'n':
            return '\n'
        elif char == 'r':
            return '\r'
        elif char == 't':
            return '\t'
        else:
            raise ValueError(f"Unexpected escape sequence: \\{char}" + " at " + str(stream.position() - 1))


class JsonArray(JsonParser, Iterator[Any]):
    def __init__(self):
        self.listener: Queue = Queue()

    @staticmethod
    def token() -> str:
        return '['

    def __next__(self) -> Any:
        result = self.listener.get()
        if result is None:
            raise StopIteration
        return result

    @override
    def parse(self, stream: Tokenator):
        normalised_stream = JsonNormalizer(stream)
        char = next(normalised_stream)
        if not char == JsonArray.token():
            raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
        separate = False
        while True:
            char = normalised_stream.peek()
            if char == ']':
                next(normalised_stream)
                self.listener.put(None)
                break

            if char == ',':
                if not separate:
                    raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
                next(normalised_stream)
                separate = False
            else:
                value = JsonObject.delegate(stream)
                self.listener.put(value)
                value.parse(stream)
                separate = True


class JsonNormalizer(PeekableIterator[str]):
    def __init__(self, stream: Tokenator):
        self.stream = stream

    @override
    def __next__(self) -> str:
        self.peek()
        return next(self.stream)

    @override
    def peek(self) -> str:
        while True:
            token = self.stream.peek()
            if str.isspace(token):
                next(self.stream)
                continue
            else:
                return token


class JsonObject(JsonParser):
    def __init__(self):
        self.listener: Queue = Queue()
        self.object: Dict[str, Any] = {}

    @staticmethod
    def token() -> str:
        return '{'

    def __getitem__(self, key: str) -> Any:
        if key in self.object.keys():
            return self.object[key]

        while True:
            entry = self.listener.get()
            if entry is None:
                raise KeyError(key)

            self.object[entry[0]] = entry[1]
            if key == entry[0]:
                return entry[1]

    @override
    def parse(self, stream: Tokenator):
        try:
            normalised_stream = JsonNormalizer(stream)
            char = next(normalised_stream)
            if not char == JsonObject.token():
                raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
            separate = False
            while True:
                char = normalised_stream.peek()

                if char == '}':
                    next(normalised_stream)
                    self.listener.put(None)
                    return

                if char == ',':
                    if not separate:
                        raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
                    next(normalised_stream)
                    separate = False
                elif char == '"':
                    if separate:
                        raise Exception(f"Unexpected symbol: {char} at {stream.position()}")

                    key = ''.join(JsonString.read(stream))
                    assert next(normalised_stream) == ':'
                    value = JsonObject.delegate(stream)
                    self.listener.put((key, value))
                    value.parse(stream)
                    separate = True
                else:
                    raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
        except StopIteration:
            return

    @staticmethod
    def delegate(stream: Tokenator):
        normalised_stream = JsonNormalizer(stream)
        char = normalised_stream.peek()
        if char == JsonObject.token():
            return JsonObject()
        if char == JsonString.token():
            return JsonString()
        if char == JsonArray.token():
            return JsonArray()
        raise Exception(f"Unexpected symbol: {char} at {stream.position()}")
