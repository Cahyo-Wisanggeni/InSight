"""Core RAG engine orchestrating parsing, indexing, retrieval, LLM synthesis, and history."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from insight.config import Settings
from insight.llm import LazyLLM
from insight.parser import chunk_markdown, parse_file
from insight.storage import DocumentRecord, MessageRecord, SessionSummary, SQLiteStore
from insight.vectorstore import LocalVectorStore, RetrievedChunk


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[dict[str, Any]]
    session_id: str


class InSightEngine:
    """Stateful workspace engine combining SQLite, Qdrant Local, Docling, and LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_dirs()
        self.store = SQLiteStore(self.settings.sqlite_path)
        self.vectors = LocalVectorStore(
            qdrant_path=self.settings.qdrant_path,
            embed_model=self.settings.embed_model,
            use_reranker=self.settings.use_reranker,
            rerank_model=self.settings.rerank_model,
        )
        self.llm = LazyLLM(self.settings)

    def ingest_file(self, file_path: Path, display_name: str | None = None) -> DocumentRecord:
        filename = display_name or file_path.name
        parsed = parse_file(file_path)
        if not parsed.markdown:
            raise ValueError(f"No readable content extracted from '{filename}'.")

        chunks = chunk_markdown(
            parsed.markdown,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"Document '{filename}' produced zero text chunks.")

        doc_id = uuid.uuid4().hex[:12]
        self.vectors.upsert_chunks(doc_id=doc_id, filename=filename, chunks=chunks)
        return self.store.add_document(
            doc_id=doc_id,
            filename=filename,
            file_ext=file_path.suffix.lower() or ".txt",
            chunk_count=len(chunks),
            char_count=len(parsed.markdown),
            parser_used=parsed.parser_used,
        )

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        record = self.store.get_document(doc_id)
        if record is None:
            return False
        self.vectors.delete_document(doc_id)
        self.store.delete_document(doc_id)
        uploaded = self.settings.upload_dir / f"{doc_id}_{record.filename}"
        if uploaded.exists():
            uploaded.unlink()
        return True

    def ask(self, question: str, session_id: str = "default") -> ChatResult:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty.")

        self.store.add_message(session_id=session_id, role="user", content=cleaned)

        docs = self.store.list_documents()
        if not docs:
            answer = "Belum ada dokumen yang diunggah. Silakan unggah dokumen terlebih dahulu."
            self.store.add_message(session_id=session_id, role="assistant", content=answer, sources=[])
            return ChatResult(answer=answer, sources=[], session_id=session_id)

        retrieved: list[RetrievedChunk] = self.vectors.search(
            query=cleaned,
            top_k_candidates=self.settings.top_k_candidates,
            top_k_final=self.settings.top_k_final,
        )
        sources = [
            {
                "doc_id": item.doc_id,
                "filename": item.filename,
                "section": item.section,
                "chunk_index": item.chunk_index,
                "score": round(item.score, 4),
                "excerpt": item.text[:220],
            }
            for item in retrieved
        ]

        answer = self.llm.generate(question=cleaned, contexts=retrieved)
        self.store.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        return ChatResult(answer=answer, sources=sources, session_id=session_id)

    def get_history(self, session_id: str = "default") -> list[MessageRecord]:
        return self.store.list_messages(session_id=session_id)

    def clear_history(self, session_id: str = "default") -> int:
        return self.store.clear_messages(session_id=session_id)

    def list_sessions(self) -> list[SessionSummary]:
        return self.store.list_sessions()
