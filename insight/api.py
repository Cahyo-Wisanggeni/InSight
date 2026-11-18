"""FastAPI backend serving the InSight RAG API and built-in chatbot web UI."""

from __future__ import annotations

import hmac
import secrets
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from insight import __version__
from insight.config import Settings
from insight.rag import InSightEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LoginRequest(BaseModel):
    password: str


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


def create_app(engine: InSightEngine | None = None) -> FastAPI:
    app = FastAPI(title="InSight", version=__version__)
    rag = engine or InSightEngine(Settings.from_env())
    active_tokens: set[str] = set()

    def require_auth(x_insight_token: str | None = Header(default=None)) -> None:
        if not rag.settings.password:
            return
        if not x_insight_token or x_insight_token not in active_tokens:
            raise HTTPException(status_code=401, detail="Authentication required.")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        docs = rag.list_documents()
        return {
            "status": "ok",
            "version": __version__,
            "auth_required": bool(rag.settings.password),
            "llm_backend": rag.settings.llm_backend,
            "llm_model": rag.settings.llm_model,
            "precision_mode": rag.settings.precision_mode,
            "embedder": rag.vectors.embedder.name,
            "use_reranker": rag.settings.use_reranker,
            "document_count": len(docs),
        }

    @app.post("/auth/login")
    def login(payload: LoginRequest) -> dict:
        if not rag.settings.password:
            return {"auth_required": False, "token": ""}
        if not hmac.compare_digest(payload.password, rag.settings.password):
            raise HTTPException(status_code=401, detail="Invalid password.")
        token = secrets.token_urlsafe(24)
        active_tokens.add(token)
        return {"auth_required": True, "token": token}

    @app.get("/documents", dependencies=[Depends(require_auth)])
    def list_documents() -> dict:
        return {"documents": [asdict(doc) for doc in rag.list_documents()]}

    @app.post("/documents/upload", dependencies=[Depends(require_auth)])
    async def upload_document(file: UploadFile = File(...)) -> dict:
        filename = Path(file.filename or "document.txt").name
        suffix = Path(filename).suffix.lower()
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=rag.settings.upload_dir,
            ) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            record = rag.ingest_file(tmp_path, display_name=filename)
            return {"document": asdict(record)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @app.delete("/documents/{doc_id}", dependencies=[Depends(require_auth)])
    def delete_document(doc_id: str) -> dict:
        deleted = rag.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"deleted": True, "doc_id": doc_id}

    @app.post("/chat", dependencies=[Depends(require_auth)])
    def chat(payload: ChatRequest) -> dict:
        try:
            result = rag.ask(question=payload.question, session_id=payload.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "answer": result.answer,
            "sources": result.sources,
            "session_id": result.session_id,
        }

    @app.get("/history", dependencies=[Depends(require_auth)])
    def get_history(session_id: str = "default") -> dict:
        messages = rag.get_history(session_id=session_id)
        return {"session_id": session_id, "messages": [asdict(m) for m in messages]}

    @app.delete("/history", dependencies=[Depends(require_auth)])
    def clear_history(session_id: str = "default") -> dict:
        deleted_count = rag.clear_history(session_id=session_id)
        return {"session_id": session_id, "deleted_count": deleted_count}

    return app
