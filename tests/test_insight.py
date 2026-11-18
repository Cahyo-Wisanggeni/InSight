"""End-to-end tests for InSight storage, parser, Qdrant vector store, and FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from insight.api import create_app
from insight.config import Settings
from insight.parser import chunk_markdown, parse_file
from insight.rag import InSightEngine


def _make_settings(tmp_path: Path, password: str = "") -> Settings:
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "insight.sqlite",
        qdrant_path=tmp_path / "qdrant_db",
        upload_dir=tmp_path / "uploads",
        llm_backend="retrieval_only",
        llm_model="test-preview",
        precision_mode="4bit",
        max_seq_length=2048,
        max_new_tokens=256,
        ollama_url="http://localhost:11434",
        embed_model="hashing-fallback",
        use_reranker=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_candidates=8,
        top_k_final=3,
        chunk_size=300,
        chunk_overlap=50,
        password=password,
    )


def test_parse_and_chunk_markdown(tmp_path: Path) -> None:
    sample = tmp_path / "notes.md"
    sample.write_text(
        "# Arsitektur InSight\n\n"
        "InSight memakai Qdrant lokal dan SQLite untuk menjaga privasi dokumen.\n\n"
        "## Penyimpanan\n\n"
        "Riwayat percakapan disimpan di SQLite agar ringan dan mudah dicari.",
        encoding="utf-8",
    )
    parsed = parse_file(sample)
    assert parsed.parser_used == "native_text"
    chunks = chunk_markdown(parsed.markdown, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].section == "Arsitektur InSight"


def test_rag_ingest_ask_history_and_delete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    settings = _make_settings(tmp_path)
    engine = InSightEngine(settings)
    app = create_app(engine)
    client = TestClient(app)

    ui_res = client.get("/")
    assert ui_res.status_code == 200
    assert "InSight" in ui_res.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["document_count"] == 0

    upload = client.post(
        "/documents/upload",
        files={"file": ("panduan.md", b"# Privasi\n\nData diproses secara lokal memakai Qdrant dan SQLite.", "text/markdown")},
    )
    assert upload.status_code == 200
    doc = upload.json()["document"]
    doc_id = doc["doc_id"]

    docs_res = client.get("/documents")
    assert docs_res.status_code == 200
    assert len(docs_res.json()["documents"]) == 1

    chat_res = client.post("/chat", json={"question": "Apa database yang dipakai untuk privasi?"})
    assert chat_res.status_code == 200
    body = chat_res.json()
    assert "panduan.md" in body["answer"]
    assert len(body["sources"]) == 1

    history_res = client.get("/history")
    assert history_res.status_code == 200
    assert len(history_res.json()["messages"]) == 2

    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert client.get("/documents").json()["documents"] == []

    # After dropping the document, Qdrant should no longer return its chunks.
    after_drop = client.post("/chat", json={"question": "Qdrant"})
    assert after_drop.status_code == 200
    assert after_drop.json()["sources"] == []

    clear_res = client.delete("/history")
    assert clear_res.status_code == 200
    assert client.get("/history").json()["messages"] == []


def test_password_protection_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    settings = _make_settings(tmp_path, password="rahasia-colab")
    app = create_app(InSightEngine(settings))
    client = TestClient(app)

    unauth = client.get("/documents")
    assert unauth.status_code == 401

    bad_login = client.post("/auth/login", json={"password": "salah"})
    assert bad_login.status_code == 401

    ok_login = client.post("/auth/login", json={"password": "rahasia-colab"})
    assert ok_login.status_code == 200
    token = ok_login.json()["token"]

    auth_docs = client.get("/documents", headers={"X-InSight-Token": token})
    assert auth_docs.status_code == 200
