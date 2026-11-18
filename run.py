"""Local & Colab entrypoint for running the InSight FastAPI server."""

from __future__ import annotations

import os
import threading
import time
import uvicorn

from insight.api import create_app


def serve(host: str, port: int, app=None) -> None:
    target_app = app or create_app()
    config = uvicorn.Config(target_app, host=host, port=port)
    server = uvicorn.Server(config)
    tracker = getattr(target_app.state, "tracker", None)

    if tracker is not None and tracker.enabled:
        def _watch_shutdown() -> None:
            while not server.should_exit:
                if tracker.should_shutdown():
                    server.should_exit = True
                    break
                time.sleep(1.0)

        threading.Thread(target=_watch_shutdown, daemon=True).start()

    server.run()


def main() -> None:
    host = os.getenv("INSIGHT_HOST", "127.0.0.1")
    port = int(os.getenv("INSIGHT_PORT", "8000"))
    serve(host=host, port=port)


if __name__ == "__main__":
    main()
