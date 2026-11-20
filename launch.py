"""One-click local launcher: starts InSight runtime, opens the UI, and stops on UI close."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _find_open_port(preferred: int = 8000) -> int:
    for candidate in [preferred, *range(preferred + 1, preferred + 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _find_chromium_app_browser() -> str | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for item in candidates:
        if item and Path(item).exists():
            return item
    return None


def _open_ui(url: str) -> None:
    browser = _find_chromium_app_browser()
    if browser:
        subprocess.Popen([browser, f"--app={url}"])
        return
    webbrowser.open(url)


def main() -> None:
    from insight.config import load_env_file

    load_env_file()
    os.environ.setdefault("INSIGHT_AUTO_SHUTDOWN", "1")
    host = os.getenv("INSIGHT_HOST", "127.0.0.1")
    preferred_port = int(os.getenv("INSIGHT_PORT", "8000"))

    from run import serve, stop_stale_instances

    stop_stale_instances(preferred_port)
    port = _find_open_port(preferred_port)
    url = f"http://{host}:{port}"

    server_thread = threading.Thread(target=serve, args=(host, port), daemon=False)
    server_thread.start()

    ready = False
    for _ in range(240):
        if not server_thread.is_alive():
            break
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.25)

    if ready:
        _open_ui(url)
    try:
        server_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
