"""Local entrypoint for running the InSight FastAPI server."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import uvicorn

from insight.api import create_app
from insight.config import load_env_file

WORKSPACE_ROOT = str(Path(__file__).resolve().parent).lower()


def stop_stale_instances(port: int) -> None:
    """Stop earlier InSight run.py/launch.py instances holding the port or Qdrant lock."""
    try:
        import psutil
    except ImportError:
        return

    current_pid = os.getpid()
    current_ppid = os.getppid()
    stale = []

    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            pid = proc.info["pid"]
            if pid in (current_pid, current_ppid):
                continue
            name = (proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            cwd = (proc.info.get("cwd") or "").lower()
            same_workspace = WORKSPACE_ROOT in cmdline or WORKSPACE_ROOT == cwd
            is_insight_script = "run.py" in cmdline or "launch.py" in cmdline
            holds_port = False
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr and conn.laddr.port == port:
                        holds_port = True
                        break
            except Exception:
                pass
            if (same_workspace and is_insight_script) or holds_port:
                stale.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for proc in stale:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if stale:
        _, alive = psutil.wait_procs(stale, timeout=3.0)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if alive:
            psutil.wait_procs(alive, timeout=1.0)


def serve(host: str, port: int, app=None) -> None:
    if app is None:
        stop_stale_instances(port)
        from insight.config import Settings
        from insight.llm import ensure_gguf_model, ensure_ollama_running

        cfg = Settings.from_env()
        ensure_gguf_model(cfg)
        ensure_ollama_running(cfg)
    target_app = app or create_app()
    config = uvicorn.Config(target_app, host=host, port=port)
    server = uvicorn.Server(config)
    tracker = getattr(target_app.state, "tracker", None)

    if tracker is not None:
        def _watch_shutdown() -> None:
            while not server.should_exit:
                if tracker.should_shutdown():
                    server.should_exit = True
                    break
                time.sleep(0.5)

        threading.Thread(target=_watch_shutdown, daemon=True).start()

    server.run()


def main() -> None:
    load_env_file()
    host = os.getenv("INSIGHT_HOST", "127.0.0.1")
    port = int(os.getenv("INSIGHT_PORT", "8000"))
    serve(host=host, port=port)


if __name__ == "__main__":
    main()
