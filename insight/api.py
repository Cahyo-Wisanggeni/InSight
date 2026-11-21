"""FastAPI backend serving the InSight RAG API and built-in chatbot web UI."""

from __future__ import annotations

import hmac
import json
import secrets
import tempfile
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from insight import __version__
from insight.config import (
    DEFAULT_RESOURCES_DIR,
    Settings,
    load_env_file,
    normalize_rag_mode,
    summarize_resources,
    summarize_resources_cached,
)
from insight.rag import VALID_RAG_MODES, InSightEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LoginRequest(BaseModel):
    password: str


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    rag_mode: str | None = None
    doc_id: str | None = None
    project_id: str | None = None
    sampling: dict[str, float | int] | None = None


class ModelSelectRequest(BaseModel):
    model_id: str


class LoginRateLimiter:
    """In-memory sliding-window rate limiter for failed login attempts."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 60) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def is_blocked(self, key: str, now: float | None = None) -> bool:
        with self._lock:
            t = time.monotonic() if now is None else now
            bucket = self._failures[key]
            self._prune(bucket, t)
            return len(bucket) >= self.max_attempts

    def record_failure(self, key: str, now: float | None = None) -> None:
        with self._lock:
            t = time.monotonic() if now is None else now
            bucket = self._failures[key]
            self._prune(bucket, t)
            bucket.append(t)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class RuntimeTracker:
    """Tracks active UI heartbeats and triggers graceful auto-shutdown on disconnect."""

    def __init__(
        self,
        enabled: bool = False,
        grace_seconds: float = 4.0,
        idle_timeout_seconds: float = 25.0,
    ) -> None:
        self.enabled = enabled
        self.grace_seconds = grace_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self._lock = threading.Lock()
        self.has_client = False
        self.last_seen: float | None = None
        self.disconnect_deadline: float | None = None
        self.manual_shutdown = False
        self.active_requests = 0

    def begin_request(self) -> None:
        with self._lock:
            self.active_requests += 1

    def end_request(self, now: float | None = None) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            if self.has_client:
                self.last_seen = time.monotonic() if now is None else now

    def request_shutdown(self) -> None:
        with self._lock:
            self.manual_shutdown = True

    def record_heartbeat(self, now: float | None = None) -> None:
        with self._lock:
            t = time.monotonic() if now is None else now
            self.has_client = True
            self.last_seen = t
            self.disconnect_deadline = None

    def record_disconnect(self, now: float | None = None) -> None:
        with self._lock:
            if not self.enabled or not self.has_client:
                return
            t = time.monotonic() if now is None else now
            self.disconnect_deadline = t + self.grace_seconds

    def should_shutdown(self, now: float | None = None) -> bool:
        with self._lock:
            if self.manual_shutdown:
                return True
            if not self.enabled or not self.has_client:
                return False
            if self.active_requests > 0:
                return False
            t = time.monotonic() if now is None else now
            if self.disconnect_deadline is not None and t >= self.disconnect_deadline:
                return True
            if self.last_seen is not None and (t - self.last_seen) >= self.idle_timeout_seconds:
                return True
            return False


def create_app(engine: InSightEngine | None = None, settings: Settings | None = None) -> FastAPI:
    if engine is not None:
        rag = engine
        app_settings = engine.settings
        llm = engine.llm
        owns_engine = False
    else:
        if settings is None:
            load_env_file()
        app_settings = settings or Settings.from_env()
        owns_engine = True
        rag = InSightEngine(app_settings)
        llm = rag.llm

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if owns_engine:
                rag.close()

    app = FastAPI(title="InSight", version=__version__, lifespan=_lifespan)

    active_tokens: dict[str, float] = {}
    token_lock = threading.Lock()
    rate_limiter = LoginRateLimiter(
        max_attempts=app_settings.login_max_attempts,
        window_seconds=app_settings.login_window_seconds,
    )
    tracker = RuntimeTracker(
        enabled=app_settings.auto_shutdown,
        grace_seconds=app_settings.shutdown_grace_seconds,
        idle_timeout_seconds=app_settings.heartbeat_timeout_seconds,
    )
    app.state.tracker = tracker

    def _is_valid_token(token: str | None) -> bool:
        if not token:
            return False
        now = time.monotonic()
        with token_lock:
            expires_at = active_tokens.get(token)
            if expires_at is None:
                return False
            if now > expires_at:
                active_tokens.pop(token, None)
                return False
            return True

    def _issue_token() -> str:
        now = time.monotonic()
        token = secrets.token_urlsafe(24)
        with token_lock:
            expired = [k for k, exp in active_tokens.items() if now > exp]
            for k in expired:
                active_tokens.pop(k, None)
            active_tokens[token] = now + app_settings.token_ttl_seconds
        return token

    def require_auth(x_insight_token: str | None = Header(default=None)) -> None:
        if not app_settings.password:
            return
        if not _is_valid_token(x_insight_token):
            raise HTTPException(status_code=401, detail="Authentication required.")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict:
        docs = rag.list_documents()
        res_summary = summarize_resources_cached(app_settings.resources_dir or DEFAULT_RESOURCES_DIR)
        return {
            "status": "ok",
            "version": __version__,
            "role": "local",
            "auth_required": bool(app_settings.password),
            "llm_backend": llm.active_backend,
            "llm_model": llm.active_model,
            "active_model_id": rag.active_model_id,
            "llm_runtime": llm.active_runtime,
            "precision_mode": llm.active_precision,
            "embedder": rag.vectors.embedder.name,
            "embedder_fallback": rag.vectors.is_fallback_embedder,
            "embedder_warning": rag.vectors.embedder_warning,
            "use_reranker": app_settings.use_reranker,
            "rag_mode": app_settings.rag_mode,
            "available_rag_modes": list(VALID_RAG_MODES),
            "document_count": len(docs),
            "resources_dir": res_summary["resources_dir"],
            "resources_mb": res_summary["total_mb"],
            "resources_files": res_summary["file_count"],
            "auto_shutdown": app_settings.auto_shutdown,
        }

    @app.get("/resources", dependencies=[Depends(require_auth)])
    def list_resources() -> dict:
        return summarize_resources(app_settings.resources_dir or DEFAULT_RESOURCES_DIR)

    @app.get("/models", dependencies=[Depends(require_auth)])
    def get_models() -> dict:
        models = rag.list_models()
        return {
            "active_model_id": rag.active_model_id,
            "models": models,
        }

    @app.post("/models/select", dependencies=[Depends(require_auth)])
    async def select_model(payload: ModelSelectRequest) -> dict:
        try:
            res = await run_in_threadpool(rag.switch_model, payload.model_id)
            return {"ok": True, "model": res}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runtime/heartbeat", dependencies=[Depends(require_auth)])
    def runtime_heartbeat() -> dict:
        tracker.record_heartbeat()
        return {"ok": True, "auto_shutdown": tracker.enabled}

    @app.post("/runtime/disconnect")
    def runtime_disconnect(
        x_insight_token: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> dict:
        if app_settings.password and not _is_valid_token(x_insight_token or token):
            raise HTTPException(status_code=401, detail="Authentication required.")
        tracker.record_disconnect()
        return {"ok": True, "auto_shutdown": tracker.enabled}

    @app.post("/runtime/shutdown", dependencies=[Depends(require_auth)])
    def runtime_shutdown() -> dict:
        tracker.request_shutdown()
        return {"ok": True, "shutting_down": True}

    @app.post("/auth/login")
    def login(payload: LoginRequest, request: Request) -> dict:
        if not app_settings.password:
            return {"auth_required": False, "token": ""}
        client_ip = request.client.host if request.client else "unknown"
        if rate_limiter.is_blocked(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many failed login attempts. Try again later.",
            )
        if not hmac.compare_digest(payload.password, app_settings.password):
            rate_limiter.record_failure(client_ip)
            raise HTTPException(status_code=401, detail="Invalid password.")
        rate_limiter.reset(client_ip)
        token = _issue_token()
        return {"auth_required": True, "token": token, "expires_in": app_settings.token_ttl_seconds}

    @app.get("/documents", dependencies=[Depends(require_auth)])
    def list_documents() -> dict:
        return {"documents": [asdict(doc) for doc in rag.list_documents()]}

    @app.get("/projects", dependencies=[Depends(require_auth)])
    def list_projects() -> dict:
        return {"projects": [asdict(p) for p in rag.list_projects()]}

    @app.post("/projects", dependencies=[Depends(require_auth)])
    def create_project(payload: dict) -> dict:
        raw_name = str((payload or {}).get("name", "")).strip()
        if not raw_name:
            raise HTTPException(status_code=400, detail="Project name cannot be empty.")
        try:
            project = rag.create_project(raw_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"project": asdict(project)}

    @app.patch("/projects/{project_id}", dependencies=[Depends(require_auth)])
    def rename_project(project_id: str, payload: dict) -> dict:
        raw_name = str((payload or {}).get("name", "")).strip()
        if not raw_name:
            raise HTTPException(status_code=400, detail="Project name cannot be empty.")
        try:
            updated = rag.rename_project(project_id, raw_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return {"project": asdict(updated)}

    @app.delete("/projects/{project_id}", dependencies=[Depends(require_auth)])
    def delete_project(project_id: str) -> dict:
        deleted = rag.delete_project(project_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Project not found.")
        return {"deleted": True, "project_id": project_id}

    @app.post("/documents/upload", dependencies=[Depends(require_auth)])
    async def upload_document(
        file: UploadFile = File(...),
        project_id: str | None = Form(None),
    ) -> dict:
        filename = Path(file.filename or "document.txt").name
        suffix = Path(filename).suffix.lower()
        max_bytes = rag.settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File terlalu besar (batas {rag.settings.max_upload_mb} MB).",
            )

        tmp_path: Path | None = None
        tracker.begin_request()
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=rag.settings.upload_dir,
            ) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            record = await run_in_threadpool(
                rag.ingest_file, tmp_path, filename, project_id=project_id or None
            )
            return {"document": asdict(record)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            tracker.end_request()
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @app.delete("/documents/{doc_id}", dependencies=[Depends(require_auth)])
    def delete_document(doc_id: str) -> dict:
        deleted = rag.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"deleted": True, "doc_id": doc_id}

    @app.patch("/documents/{doc_id}", dependencies=[Depends(require_auth)])
    def rename_document(doc_id: str, payload: dict) -> dict:
        body = payload or {}
        updated = None
        if "filename" in body:
            raw_name = str(body.get("filename", "")).strip()
            if not raw_name:
                raise HTTPException(status_code=400, detail="Document name cannot be empty.")
            try:
                updated = rag.rename_document(doc_id, raw_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if updated is None:
                raise HTTPException(status_code=404, detail="Document not found.")
        if "project_id" in body:
            raw_proj = body.get("project_id")
            proj_id = str(raw_proj).strip() if raw_proj is not None else None
            updated = rag.set_document_project(doc_id, proj_id or None)
            if updated is None:
                raise HTTPException(status_code=404, detail="Document not found.")
        if updated is None:
            raise HTTPException(status_code=400, detail="No document fields provided to update.")
        return {"document": asdict(updated)}

    @app.post("/chat", dependencies=[Depends(require_auth)])
    def chat(payload: ChatRequest) -> dict:
        tracker.begin_request()
        try:
            result = rag.ask(
                question=payload.question,
                session_id=payload.session_id,
                rag_mode=payload.rag_mode,
                doc_id=payload.doc_id,
                project_id=payload.project_id,
                sampling_params=payload.sampling,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            tracker.end_request()
        return {
            "answer": result.answer,
            "sources": result.sources,
            "session_id": result.session_id,
            "rag_mode": result.rag_mode,
            "retrieval_strategy": result.retrieval_strategy,
            "query_analysis": result.query_analysis,
        }

    @app.post("/chat/stream", dependencies=[Depends(require_auth)])
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        cleaned = payload.question.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        try:
            if payload.rag_mode is not None:
                normalize_rag_mode(payload.rag_mode, default=rag.settings.rag_mode, strict=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def _event_iter():
            tracker.begin_request()
            try:
                for event in rag.ask_stream(
                    question=cleaned,
                    session_id=payload.session_id,
                    rag_mode=payload.rag_mode,
                    doc_id=payload.doc_id,
                    project_id=payload.project_id,
                    sampling_params=payload.sampling,
                ):
                    evt_type = event.get("type", "message")
                    yield f"event: {evt_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                tracker.end_request()

        return StreamingResponse(
            _event_iter(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/history", dependencies=[Depends(require_auth)])
    def get_history(session_id: str = "default") -> dict:
        messages = rag.get_history(session_id=session_id)
        return {"session_id": session_id, "messages": [asdict(m) for m in messages]}

    @app.delete("/history", dependencies=[Depends(require_auth)])
    def clear_history(session_id: str = "default", all: bool = False) -> dict:
        if all:
            deleted_count = rag.clear_all_history()
            return {"all": True, "deleted_count": deleted_count}
        deleted_count = rag.clear_history(session_id=session_id)
        return {"session_id": session_id, "deleted_count": deleted_count}

    @app.get("/sessions", dependencies=[Depends(require_auth)])
    def list_sessions() -> dict:
        sessions = rag.list_sessions()
        return {"sessions": [asdict(s) for s in sessions]}

    @app.patch("/sessions/{session_id}", dependencies=[Depends(require_auth)])
    def rename_session(session_id: str, payload: dict) -> dict:
        title = str((payload or {}).get("title", "")).strip()
        if not title:
            raise HTTPException(status_code=400, detail="Conversation title cannot be empty.")
        try:
            renamed = rag.rename_session(session_id=session_id, title=title)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not renamed:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"session_id": session_id, "title": title}

    return app
