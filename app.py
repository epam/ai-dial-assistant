#!/usr/bin/env python3

import uvicorn
from aidial_sdk import (
    DIALApp,
)
from starlette.responses import Response, FileResponse

from application.assistant_application import AssistantApplication


app = DIALApp()
app.add_chat_completion("assistant", AssistantApplication())


@app.get("/healthcheck/status200")
def status200() -> Response:
    return Response("Service is running...", status_code=200)


@app.get("/{plugin}/.well-known/{filename}")
def read_file(plugin: str, filename: str):
    return FileResponse(f"{plugin}/.well-known/{filename}")


def main():
    uvicorn.run(app, port=7001)


if __name__ == "__main__":
    main()
