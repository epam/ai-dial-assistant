import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Addon framework port", default=5000)

    args = parser.parse_args()
    uvicorn.run(
        "aidial_assistant.app:app", port=int(args.port), env_file=".env"
    )


if __name__ == "__main__":
    main()
