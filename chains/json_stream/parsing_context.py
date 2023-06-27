from asyncio import Task

from chains.json_stream.json_node import JsonNode


class ParsingContext:
    def __init__(self, node: JsonNode, task: Task):
        self._node = node
        self._task = task

    @property
    def node(self) -> JsonNode:
        return self._node

    async def finish_parsing(self):
        await self._task

