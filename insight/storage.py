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
class ProjectRecord:
    project_id: str
    name: str
    created_at: str
    doc_count: int = 0


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    file_ext: str
    chunk_count: int
    char_count: int
    parser_used: str
    created_at: str
    project_id: str | None = None


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
    doc_id: str | None = None
    project_id: str | None = None


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
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    parser_used TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    project_id TEXT
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

                CREATE TABLE IF NOT EXISTS session_titles (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    doc_id TEXT,
                    project_id TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
            doc_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "project_id" not in doc_cols:
                conn.execute("ALTER TABLE documents ADD COLUMN project_id TEXT")
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(session_titles)").fetchall()
            }
            if "doc_id" not in cols:
                conn.execute("ALTER TABLE session_titles ADD COLUMN doc_id TEXT")
            if "project_id" not in cols:
                conn.execute("ALTER TABLE session_titles ADD COLUMN project_id TEXT")

    def add_document(
        self,
        doc_id: str,
        filename: str,
        file_ext: str,
        chunk_count: int,
        char_count: int,
        parser_used: str,
        project_id: str | None = None,
    ) -> DocumentRecord:
        created_at = _utc_now()
        clean_proj = project_id.strip() if project_id else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at, clean_proj),
            )
        return DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            file_ext=file_ext,
            chunk_count=chunk_count,
            char_count=char_count,
            parser_used=parser_used,
            created_at=created_at,
            project_id=clean_proj,
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at, project_id
                FROM documents
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at, project_id
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def find_documents_by_filename(
        self,
        filename: str,
        project_id: str | None = None,
    ) -> list[DocumentRecord]:
        clean_proj = project_id.strip() if project_id else None
        with self._connect() as conn:
            if clean_proj is None:
                rows = conn.execute(
                    """
                    SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at, project_id
                    FROM documents
                    WHERE filename = ? AND project_id IS NULL
                    ORDER BY created_at DESC
                    """,
                    (filename,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT doc_id, filename, file_ext, chunk_count, char_count, parser_used, created_at, project_id
                    FROM documents
                    WHERE filename = ? AND project_id = ?
                    ORDER BY created_at DESC
                    """,
                    (filename, clean_proj),
                ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def set_document_project(self, doc_id: str, project_id: str | None) -> DocumentRecord | None:
        clean_proj = project_id.strip() if project_id else None
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE documents SET project_id = ? WHERE doc_id = ?",
                (clean_proj, doc_id),
            )
        if cur.rowcount == 0:
            return None
        return self.get_document(doc_id)

    def list_project_doc_ids(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        return [str(row["doc_id"]) for row in rows]

    def create_project(self, project_id: str, name: str) -> ProjectRecord:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Project name cannot be empty.")
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects (project_id, name, created_at) VALUES (?, ?, ?)",
                (project_id, cleaned, created_at),
            )
        return ProjectRecord(project_id=project_id, name=cleaned, created_at=created_at, doc_count=0)

    def list_projects(self) -> list[ProjectRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.project_id,
                    p.name,
                    p.created_at,
                    COUNT(d.doc_id) AS doc_count
                FROM projects p
                LEFT JOIN documents d ON d.project_id = p.project_id
                GROUP BY p.project_id
                ORDER BY p.created_at DESC
                """
            ).fetchall()
        return [ProjectRecord(**dict(row)) for row in rows]

    def rename_project(self, project_id: str, name: str) -> ProjectRecord | None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Project name cannot be empty.")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE projects SET name = ? WHERE project_id = ?",
                (cleaned, project_id),
            )
            if cur.rowcount == 0:
                return None
            row = conn.execute(
                """
                SELECT p.project_id, p.name, p.created_at, COUNT(d.doc_id) AS doc_count
                FROM projects p
                LEFT JOIN documents d ON d.project_id = p.project_id
                WHERE p.project_id = ?
                GROUP BY p.project_id
                """,
                (project_id,),
            ).fetchone()
        return ProjectRecord(**dict(row)) if row else None

    def delete_project(self, project_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            conn.execute("UPDATE documents SET project_id = NULL WHERE project_id = ?", (project_id,))
            conn.execute("UPDATE session_titles SET project_id = NULL WHERE project_id = ?", (project_id,))
        return cur.rowcount > 0

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("UPDATE session_titles SET doc_id = NULL WHERE doc_id = ?", (doc_id,))
        return cur.rowcount > 0

    def rename_document(self, doc_id: str, new_filename: str) -> DocumentRecord | None:
        cleaned = new_filename.strip()
        if not cleaned:
            raise ValueError("Document name cannot be empty.")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE documents SET filename = ? WHERE doc_id = ?",
                (cleaned, doc_id),
            )
        if cur.rowcount == 0:
            return None
        return self.get_document(doc_id)

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
            conn.execute("DELETE FROM session_titles WHERE session_id = ?", (session_id,))
        return cur.rowcount

    def clear_all_messages(self) -> int:
        """Delete all chat messages and session titles across every session."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM session_titles")
        return cur.rowcount

    def rename_session(self, session_id: str, title: str) -> bool:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("Conversation title cannot be empty.")
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            if exists is None:
                return False
            conn.execute(
                """
                INSERT INTO session_titles (session_id, title, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at
                """,
                (session_id, cleaned, _utc_now()),
            )
        return True

    def set_session_doc(self, session_id: str, doc_id: str | None) -> None:
        cleaned_doc = doc_id.strip() if doc_id else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_titles (session_id, title, doc_id, updated_at)
                VALUES (?, '', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    updated_at = excluded.updated_at
                """,
                (session_id, cleaned_doc, _utc_now()),
            )

    def get_session_doc(self, session_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id FROM session_titles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["doc_id"] if row and row["doc_id"] else None

    def set_session_project(self, session_id: str, project_id: str | None) -> None:
        cleaned_proj = project_id.strip() if project_id else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_titles (session_id, title, project_id, updated_at)
                VALUES (?, '', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    updated_at = excluded.updated_at
                """,
                (session_id, cleaned_proj, _utc_now()),
            )

    def get_session_project(self, session_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT project_id FROM session_titles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["project_id"] if row and row["project_id"] else None

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    m1.session_id,
                    COALESCE(
                        NULLIF(st.title, ''),
                        (
                            SELECT m2.content
                            FROM messages m2
                            WHERE m2.session_id = m1.session_id AND m2.role = 'user'
                            ORDER BY m2.id ASC
                            LIMIT 1
                        ),
                        m1.session_id
                    ) AS title,
                    st.doc_id AS doc_id,
                    st.project_id AS project_id,
                    COUNT(*) AS message_count,
                    MAX(m1.created_at) AS updated_at
                FROM messages m1
                LEFT JOIN session_titles st ON st.session_id = m1.session_id
                GROUP BY m1.session_id
                ORDER BY MAX(m1.id) DESC
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
                doc_id=row["doc_id"],
                project_id=row["project_id"],
            )
            for row in rows
        ]
