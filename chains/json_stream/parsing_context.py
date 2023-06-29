from asyncio import Task

from chains.json_stream.json_root import JsonRoot


class ParsingContext:
    def __init__(self, node: JsonRoot, task: Task):
        self._node = node
        self._task = task

    @property
    def root(self) -> JsonRoot:
        return self._node

    async def finish_parsing(self):
        await self._task

