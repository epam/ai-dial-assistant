from typing import Any

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing_extensions import override


class CallbackWithNewLines(StreamingStdOutCallbackHandler):
    prev: str

    def __init__(self) -> None:
        super().__init__()
        self.prev = ""

    @override
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Replaces \\n with \n
        # Replaces \\" with "

        if len(token) == 0:
            return

        token = self.prev + token
        self.prev = ""

        # The escaped symbol may be split between two consecutive tokens
        if token[-1] == "\\":
            self.prev = token[-1]
            token = token[:-1]

        s = token.replace("\\n", "\n").replace('\\"', '"')
        print(s, end="")
