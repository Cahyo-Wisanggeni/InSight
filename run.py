"""Local & Colab entrypoint for running the InSight FastAPI server."""

from __future__ import annotations

import os
import uvicorn

from insight.api import create_app

app = create_app()


def main() -> None:
    host = os.getenv("INSIGHT_HOST", "127.0.0.1")
    port = int(os.getenv("INSIGHT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
