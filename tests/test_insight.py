"""End-to-end tests for InSight storage, parser, Qdrant vector store, and FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from insight.api import create_app
from insight.config import Settings
from insight.parser import chunk_markdown, parse_file
from insight.rag import InSightEngine


def _make_settings(tmp_path: Path, password: str = "", auto_shutdown: bool = False) -> Settings:
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
        auto_shutdown=auto_shutdown,
        shutdown_grace_seconds=4.0,
        heartbeat_timeout_seconds=25.0,
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

    sessions_res = client.get("/sessions")
    assert sessions_res.status_code == 200
    assert len(sessions_res.json()["sessions"]) == 1
    assert sessions_res.json()["sessions"][0]["session_id"] == "default"

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


def test_runtime_auto_shutdown_lifecycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    settings = _make_settings(tmp_path, password="secret", auto_shutdown=True)
    app = create_app(InSightEngine(settings))
    tracker = app.state.tracker
    client = TestClient(app)

    assert tracker.should_shutdown(now=100.0) is False
    assert client.post("/runtime/disconnect").status_code == 401

    token = client.post("/auth/login", json={"password": "secret"}).json()["token"]
    assert client.post("/runtime/heartbeat", headers={"X-InSight-Token": token}).status_code == 200
    assert tracker.has_client is True
    assert tracker.should_shutdown(now=100.0) is False

    # Simulate disconnect via sendBeacon token query param, then cancel via refresh heartbeat.
    assert client.post(f"/runtime/disconnect?token={token}").status_code == 200
    assert tracker.disconnect_deadline is not None
    tracker.record_heartbeat(now=101.0)
    assert tracker.disconnect_deadline is None

    # Now simulate real close past the 4s grace window.
    tracker.record_disconnect(now=102.0)
    assert tracker.should_shutdown(now=105.9) is False
    assert tracker.should_shutdown(now=106.0) is True


def _build_minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


def test_fallback_parsers_pdf_docx_pptx_xlsx_html(tmp_path: Path, monkeypatch) -> None:
    import sys
    import docx
    import openpyxl
    from pptx import Presentation

    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(_build_minimal_pdf("Hello InSight PDF"))
    parsed_pdf = parse_file(pdf_file)
    assert parsed_pdf.parser_used == "pypdfium2"
    assert "Hello InSight PDF" in parsed_pdf.markdown

    docx_file = tmp_path / "sample.docx"
    d = docx.Document()
    d.add_heading("Bab Satu", level=1)
    d.add_paragraph("Isi dokumen Word untuk InSight.")
    d.save(docx_file)
    parsed_docx = parse_file(docx_file)
    assert parsed_docx.parser_used == "python-docx"
    assert "# Bab Satu" in parsed_docx.markdown
    assert "Isi dokumen Word" in parsed_docx.markdown

    pptx_file = tmp_path / "sample.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Judul Slide"
    slide.placeholders[1].text = "Isi slide PowerPoint."
    prs.save(pptx_file)
    parsed_pptx = parse_file(pptx_file)
    assert parsed_pptx.parser_used == "python-pptx"
    assert "Judul Slide" in parsed_pptx.markdown

    xlsx_file = tmp_path / "sample.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ringkasan"
    ws.append(["Metrik", "Nilai"])
    ws.append(["Akurasi", 95])
    wb.save(xlsx_file)
    parsed_xlsx = parse_file(xlsx_file)
    assert parsed_xlsx.parser_used == "openpyxl"
    assert "## Sheet: Ringkasan" in parsed_xlsx.markdown
    assert "Akurasi | 95" in parsed_xlsx.markdown

    html_file = tmp_path / "sample.html"
    html_file.write_text("<html><script>alert(1)</script><body><h1>Laporan</h1><p>Teks HTML bersih.</p></body></html>", encoding="utf-8")
    parsed_html = parse_file(html_file)
    assert parsed_html.parser_used == "beautifulsoup4"
    assert "Teks HTML bersih." in parsed_html.markdown
    assert "alert(1)" not in parsed_html.markdown


def test_greeting_and_ollama_unreachable_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "insight.sqlite",
        qdrant_path=tmp_path / "qdrant_db",
        upload_dir=tmp_path / "uploads",
        llm_backend="ollama",
        llm_model="qwen2.5:3b",
        precision_mode="4bit",
        max_seq_length=2048,
        max_new_tokens=256,
        ollama_url="http://127.0.0.1:59999",
        embed_model="hashing-fallback",
        use_reranker=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_candidates=8,
        top_k_final=3,
        chunk_size=300,
        chunk_overlap=50,
        password="",
        auto_shutdown=False,
        shutdown_grace_seconds=4.0,
        heartbeat_timeout_seconds=25.0,
    )
    app = create_app(InSightEngine(settings))
    client = TestClient(app)

    # Greeting before any upload should give a friendly response, not fail.
    greet_empty = client.post("/chat", json={"question": "halo?"})
    assert greet_empty.status_code == 200
    assert "Halo! Saya InSight" in greet_empty.json()["answer"]

    # Upload a document, test greeting with docs, and test unreachable Ollama fallback.
    up = client.post(
        "/documents/upload",
        files={"file": ("kebijakan.md", b"# Keamanan\n\nInSight menyimpan vektor di Qdrant lokal dan riwayat di SQLite.", "text/markdown")},
    )
    assert up.status_code == 200

    greet_with_docs = client.post("/chat", json={"question": "halo"})
    assert greet_with_docs.status_code == 200
    assert "kebijakan.md" in greet_with_docs.json()["answer"]

    ask_res = client.post("/chat", json={"question": "Di mana InSight menyimpan vektor?"})
    assert ask_res.status_code == 200
    assert "Qdrant" in ask_res.json()["answer"]
    assert "kebijakan.md" in ask_res.json()["answer"]
