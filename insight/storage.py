"""SQLite storage for document registry and chat history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    file_ext: str
    chunk_count: int
    char_count: int
    parser_used: str
    created_at: str


@dataclass(frozen=True)
class MessageRecord:
    id: int
    session_id: str
    role: str
    content: str
    sources: list[dict[str, Any]]
    created_at: str


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    title: str
    message_count: int
    updated_at: str


class SQLiteStore:
    """Lightweight persistent store backed by a single SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    parser_used TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages (session_id, id);
                """
            )

    def add_document(
        self,
        doc_id: str,
        filename: str,
        file_ext: str,
        chunk_count: int,
        char_count: int,
        parser_used: str,
    ) -> DocumentRecord:
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at),
            )
        return DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            file_ext=file_ext,
            chunk_count=chunk_count,
            char_count=char_count,
            parser_used=parser_used,
            created_at=created_at,
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at
                FROM documents
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def find_documents_by_filename(self, filename: str) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at
                FROM documents
                WHERE filename = ?
                ORDER BY created_at DESC
                """,
                (filename,),
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> MessageRecord:
        created_at = _utc_now()
        payload = json.dumps(sources or [], ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, payload, created_at),
            )
            msg_id = int(cur.lastrowid)
        return MessageRecord(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            sources=sources or [],
            created_at=created_at,
        )

    def list_messages(self, session_id: str = "default", limit: int = 100) -> list[MessageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, sources_json, created_at
                FROM (
                    SELECT id, session_id, role, content, sources_json, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [
            MessageRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                sources=json.loads(row["sources_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def clear_messages(self, session_id: str = "default") -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return cur.rowcount

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    session_id,
                    COALESCE(
                        (
                            SELECT m2.content
                            FROM messages m2
                            WHERE m2.session_id = m1.session_id AND m2.role = 'user'
                            ORDER BY m2.id ASC
                            LIMIT 1
                        ),
                        session_id
                    ) AS title,
                    COUNT(*) AS message_count,
                    MAX(created_at) AS updated_at
                FROM messages m1
                GROUP BY session_id
                ORDER BY MAX(id) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"],
                title=str(row["title"])[:64],
                message_count=int(row["message_count"]),
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
