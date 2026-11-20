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
    settings = _make_settings(tmp_path, password="rahasia-lokal")
    app = create_app(InSightEngine(settings))
    client = TestClient(app)

    unauth = client.get("/documents")
    assert unauth.status_code == 401

    bad_login = client.post("/auth/login", json={"password": "salah"})
    assert bad_login.status_code == 401

    ok_login = client.post("/auth/login", json={"password": "rahasia-lokal"})
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


def test_empty_docs_and_ollama_unreachable_fallback(tmp_path: Path, monkeypatch) -> None:
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

    # Chat before any upload should prompt the user to upload a document first.
    empty_res = client.post("/chat", json={"question": "halo?"})
    assert empty_res.status_code == 200
    assert "Belum ada dokumen yang diunggah" in empty_res.json()["answer"]

    # Upload a document and test unreachable Ollama fallback.
    up = client.post(
        "/documents/upload",
        files={"file": ("kebijakan.md", b"# Keamanan\n\nInSight menyimpan vektor di Qdrant lokal dan riwayat di SQLite.", "text/markdown")},
    )
    assert up.status_code == 200

    ask_res = client.post("/chat", json={"question": "Di mana InSight menyimpan vektor?"})
    assert ask_res.status_code == 200
    assert "Qdrant" in ask_res.json()["answer"]
    assert "kebijakan.md" in ask_res.json()["answer"]


def test_login_rate_limit_and_token_ttl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    now = {"t": 1000.0}
    monkeypatch.setattr("insight.api.time.monotonic", lambda: now["t"])

    settings = Settings(
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
        password="secret",
        auto_shutdown=False,
        shutdown_grace_seconds=4.0,
        heartbeat_timeout_seconds=25.0,
        token_ttl_seconds=30,
        login_max_attempts=5,
        login_window_seconds=60,
    )
    app = create_app(InSightEngine(settings))
    client = TestClient(app)

    for _ in range(5):
        assert client.post("/auth/login", json={"password": "wrong"}).status_code == 401
    assert client.post("/auth/login", json={"password": "secret"}).status_code == 429

    # Advance past the 60s lockout window and log in successfully.
    now["t"] += 61.0
    ok = client.post("/auth/login", json={"password": "secret"})
    assert ok.status_code == 200
    token = ok.json()["token"]
    assert ok.json()["expires_in"] == 30

    assert client.get("/documents", headers={"X-InSight-Token": token}).status_code == 200

    # Advance past 30s TTL: token must be rejected on both protected routes and disconnect.
    now["t"] += 31.0
    assert client.get("/documents", headers={"X-InSight-Token": token}).status_code == 401
    assert client.post(f"/runtime/disconnect?token={token}").status_code == 401


def test_runtime_manual_shutdown_endpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )

    settings = _make_settings(tmp_path, password="secret", auto_shutdown=False)
    app = create_app(InSightEngine(settings))
    client = TestClient(app)
    tracker = app.state.tracker

    assert tracker.should_shutdown() is False
    assert client.post("/runtime/shutdown").status_code == 401

    token = client.post("/auth/login", json={"password": "secret"}).json()["token"]
    res = client.post("/runtime/shutdown", headers={"X-InSight-Token": token})
    assert res.status_code == 200
    assert res.json()["shutting_down"] is True
    assert tracker.should_shutdown() is True


def _build_multipage_pdf(page_texts: list[str]) -> bytes:
    """Build a valid minimal multi-page PDF with one text line per page."""
    objs: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Font, then pairs (Page, Content) starting at 4.
    page_obj_ids = [4 + i * 2 for i in range(len(page_texts))]
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids).encode("ascii")
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids.decode('ascii')}] /Count {len(page_texts)} >>\nendobj\n".encode("ascii"))
    objs.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    for idx, text in enumerate(page_texts):
        page_id = 4 + idx * 2
        content_id = page_id + 1
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objs.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n".encode("ascii")
        )
        objs.append(
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        )

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


def test_pdf_page_label_tracking_and_chat_sources(tmp_path: Path, monkeypatch) -> None:
    import sys

    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)

    pdf_bytes = _build_multipage_pdf([
        "Alpha halaman pertama tentang arsitektur Qdrant.",
        "Beta halaman kedua tentang keamanan token TTL.",
    ])
    pdf_path = tmp_path / "laporan.pdf"
    pdf_path.write_bytes(pdf_bytes)

    parsed = parse_file(pdf_path)
    assert parsed.parser_used == "pypdfium2"
    assert "<!-- INSIGHT_PAGE:1 -->" in parsed.markdown
    assert "<!-- INSIGHT_PAGE:2 -->" in parsed.markdown

    # Single-page chunks when chunk_size separates pages, and cross-page range when combined.
    combined_chunks = chunk_markdown(parsed.markdown, chunk_size=500, chunk_overlap=0)
    assert len(combined_chunks) == 1
    assert combined_chunks[0].page_label == "1-2"
    assert "INSIGHT_PAGE" not in combined_chunks[0].text

    split_chunks = chunk_markdown(parsed.markdown, chunk_size=60, chunk_overlap=0)
    assert [c.page_label for c in split_chunks] == ["1", "2"]

    # Non-paginated markdown must keep empty page_label.
    md_chunks = chunk_markdown("# Judul\n\nParagraf biasa tanpa halaman.", chunk_size=300, chunk_overlap=0)
    assert md_chunks[0].page_label == ""

    # End-to-end API upload -> chat -> history persistence of page_label.
    settings = _make_settings(tmp_path)
    app = create_app(InSightEngine(settings))
    client = TestClient(app)

    up = client.post(
        "/documents/upload",
        files={"file": ("laporan.pdf", pdf_bytes, "application/pdf")},
    )
    assert up.status_code == 200

    chat_res = client.post("/chat", json={"question": "Apa isi keamanan token TTL?"})
    assert chat_res.status_code == 200
    sources = chat_res.json()["sources"]
    assert len(sources) >= 1
    assert sources[0]["page_label"] in {"2", "1-2"}

    hist = client.get("/history").json()["messages"]
    assert hist[-1]["sources"][0]["page_label"] == sources[0]["page_label"]


def test_docling_pdf_page_break_markers(tmp_path: Path, monkeypatch) -> None:
    import insight.parser as parser_mod

    class FakeDoc:
        def export_to_markdown(self, page_break_placeholder: str = "") -> str:
            return f"# Bab 1\n\nTeks halaman satu.{page_break_placeholder}## Bab 2\n\nTeks halaman dua."

    class FakeResult:
        document = FakeDoc()

    class FakeConverter:
        def convert(self, _path: str) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr(parser_mod, "_get_docling_converter", lambda: FakeConverter())

    fake_pdf = tmp_path / "docling.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    parsed = parse_file(fake_pdf)
    assert parsed.parser_used == "docling"
    chunks = chunk_markdown(parsed.markdown, chunk_size=300, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].section == "Bab 1"
    assert chunks[0].page_label == "1"
    assert chunks[1].section == "Bab 2"
    assert chunks[1].page_label == "2"


def test_merge_split_markdown_tables_across_pages() -> None:
    # 1. Split table with repeated header + separator -> merged cleanly without duplicate header.
    md_repeated = (
        "<!-- INSIGHT_PAGE:2 -->\n\n"
        "| Produk | Harga |\n"
        "| --- | --- |\n"
        "| Apel | 10000 |\n\n"
        "<!-- INSIGHT_PAGE:3 -->\n\n"
        "| Produk | Harga |\n"
        "| --- | --- |\n"
        "| Jeruk | 12000 |\n"
        "| Mangga | 15000 |"
    )
    chunks = chunk_markdown(md_repeated, chunk_size=600, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page_label == "2-3"
    assert chunks[0].text.count("| Produk | Harga |") == 1
    assert "| Apel | 10000 |" in chunks[0].text
    assert "| Jeruk | 12000 |" in chunks[0].text
    assert "| Mangga | 15000 |" in chunks[0].text

    # 2. Continuation rows without repeated header -> merged cleanly.
    md_cont = (
        "<!-- INSIGHT_PAGE:4 -->\n\n"
        "| Kode | Status |\n"
        "| --- | --- |\n"
        "| A1 | Aktif |\n\n"
        "<!-- INSIGHT_PAGE:5 -->\n\n"
        "| B2 | Selesai |"
    )
    chunks_cont = chunk_markdown(md_cont, chunk_size=600, chunk_overlap=0)
    assert len(chunks_cont) == 1
    assert chunks_cont[0].page_label == "4-5"
    assert "| A1 | Aktif |" in chunks_cont[0].text
    assert "| B2 | Selesai |" in chunks_cont[0].text

    # 3. Different column counts -> NOT merged.
    md_diff_cols = (
        "<!-- INSIGHT_PAGE:1 -->\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "<!-- INSIGHT_PAGE:2 -->\n\n"
        "| X | Y | Z |\n"
        "| --- | --- | --- |\n"
        "| 3 | 4 | 5 |"
    )
    chunks_diff = chunk_markdown(md_diff_cols, chunk_size=70, chunk_overlap=0)
    assert len(chunks_diff) == 2
    assert chunks_diff[0].page_label == "1"
    assert chunks_diff[1].page_label == "2"

    # 4. Intervening prose -> NOT merged.
    md_prose = (
        "<!-- INSIGHT_PAGE:1 -->\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "Paragraf pemisah di antara dua tabel.\n\n"
        "<!-- INSIGHT_PAGE:2 -->\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 3 | 4 |"
    )
    chunks_prose = chunk_markdown(md_prose, chunk_size=90, chunk_overlap=0)
    assert len(chunks_prose) >= 2
    assert sum(c.text.count("| A | B |") for c in chunks_prose) == 2


def test_embedder_fallback_health_and_strict_mode(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from insight.vectorstore import HashingEmbedder, LocalVectorStore

    monkeypatch.setattr("insight.vectorstore.build_embedder", lambda _name: HashingEmbedder())

    settings = _make_settings(tmp_path)
    app = create_app(InSightEngine(settings))
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["embedder_fallback"] is True
    assert isinstance(health["embedder_warning"], str) and len(health["embedder_warning"]) > 0

    # Strict embedder mode via env & constructor must raise RuntimeError on HashingEmbedder.
    monkeypatch.setenv("INSIGHT_STRICT_EMBEDDER", "true")
    env_settings = Settings.from_env(base_dir=tmp_path / "strict_env")
    assert env_settings.strict_embedder is True

    with pytest.raises(RuntimeError, match="INSIGHT_STRICT_EMBEDDER"):
        LocalVectorStore(
            qdrant_path=tmp_path / "strict_qdrant",
            embed_model="hashing-fallback",
            strict_embedder=True,
        )

    # Non-fallback neural embedder reports embedder_fallback=False and embedder_warning=None.
    class FakeNeuralEmbedder:
        dimension = 8
        name = "fastembed:fake-neural"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.25] * self.dimension for _ in texts]

    monkeypatch.setattr("insight.vectorstore.build_embedder", lambda _name: FakeNeuralEmbedder())
    neural_app = create_app(InSightEngine(_make_settings(tmp_path / "neural")))
    neural_health = TestClient(neural_app).get("/health").json()
    assert neural_health["embedder_fallback"] is False
    assert neural_health["embedder_warning"] is None


def test_runtime_tracker_pauses_shutdown_during_active_requests_and_upload_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    settings = Settings(
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
        password="",
        auto_shutdown=True,
        shutdown_grace_seconds=4.0,
        heartbeat_timeout_seconds=10.0,
        max_upload_mb=1,
    )
    app = create_app(InSightEngine(settings))
    tracker = app.state.tracker
    client = TestClient(app)

    tracker.record_heartbeat(now=100.0)
    tracker.begin_request()
    # Even past idle timeout, active request prevents premature shutdown.
    assert tracker.should_shutdown(now=200.0) is False
    tracker.end_request(now=200.0)
    assert tracker.should_shutdown(now=205.0) is False
    assert tracker.should_shutdown(now=211.0) is True

    # Upload exceeding max_upload_mb (1 MB) returns HTTP 413.
    oversized = b"a" * (1024 * 1024 + 16)
    res = client.post("/documents/upload", files={"file": ("huge.txt", oversized, "text/plain")})
    assert res.status_code == 413


def test_ui_serves_page_citation_and_embedder_banner_markup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    app = create_app(InSightEngine(_make_settings(tmp_path)))
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-cache"
    assert 'id="embedder-warning-banner"' in res.text
    assert "source-page-badge" in res.text
    assert "Hal. ${src.page_label}" in res.text
    assert 'id="docs-stage-panel"' in res.text
    assert 'id="sidebar-backdrop"' in res.text
    assert "color-scheme: light;" in res.text
    assert "#5661f6" in res.text
    assert "What's in your mind?..." in res.text
    assert "Good day! How may I assist you today?" in res.text
    assert 'id="canvas-theme-group"' in res.text
    assert 'data-canvas-theme' in res.text
    assert "Learn, Discover" in res.text
    assert "colab" not in res.text.lower()


def test_reupload_same_filename_replaces_without_duplicates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    app = create_app(InSightEngine(_make_settings(tmp_path)))
    client = TestClient(app)

    first = client.post(
        "/documents/upload",
        files={"file": ("laporan.md", b"# Versi 1\n\nTarget penjualan kuartal satu adalah 100 unit.", "text/markdown")},
    )
    assert first.status_code == 200
    first_id = first.json()["document"]["doc_id"]

    second = client.post(
        "/documents/upload",
        files={"file": ("laporan.md", b"# Versi 2\n\nTarget penjualan kuartal satu direvisi menjadi 250 unit.", "text/markdown")},
    )
    assert second.status_code == 200
    second_id = second.json()["document"]["doc_id"]
    assert second_id != first_id

    docs = client.get("/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == second_id

    chat = client.post("/chat", json={"question": "Berapa target penjualan kuartal satu?"})
    assert chat.status_code == 200
    assert "250 unit" in chat.json()["answer"]
    assert len(chat.json()["sources"]) == 1


def test_hybrid_and_adaptive_rag_modes_and_strict_two_options(tmp_path: Path, monkeypatch) -> None:
    import re
    import pytest
    from insight.config import normalize_rag_mode
    from insight.rag import QueryAnalyzer
    from insight.vectorstore import _bm25_scores, _normalize_scores

    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )

    # Verify BM25 & normalization keep positive score on single-document corpus.
    single_bm25 = _bm25_scores("qdrant hybrid", ["Qdrant hybrid vector search"])
    assert len(single_bm25) == 1 and single_bm25[0] > 0.0
    assert _normalize_scores(single_bm25, preserve_positive_unit=False) == [1.0]

    # Verify QueryAnalyzer + AdaptiveRAG 4-strategy routing.
    analyzer = QueryAnalyzer()
    simple_q = analyzer.analyze_query("Apa itu Qdrant?")
    assert simple_q["complexity"] == "simple"
    assert simple_q["is_factual"] is True
    assert InSightEngine.select_adaptive_strategy(simple_q) == "direct_llm"

    medium_q = analyzer.analyze_query("How does hybrid search work in documents?")
    assert medium_q["complexity"] == "medium"
    assert InSightEngine.select_adaptive_strategy(medium_q) == "hybrid_search"

    complex_q = analyzer.analyze_query(
        "Compare HybridRAG versus AdaptiveRAG for complex multi-page PDF documents"
    )
    assert complex_q["needs_multiple_sources"] is True
    assert InSightEngine.select_adaptive_strategy(complex_q) == "comprehensive"

    # Verify strict mode restriction to only HybridRAG and AdaptiveRAG.
    assert normalize_rag_mode("HybridRAG", strict=True) == "hybrid"
    assert normalize_rag_mode("AdaptiveRAG", strict=True) == "adaptive"
    with pytest.raises(ValueError, match="Hanya tersedia 2 opsi"):
        normalize_rag_mode("graph_rag", strict=True)

    monkeypatch.setenv("INSIGHT_RAG_MODE", "adaptive")
    env_settings = Settings.from_env(base_dir=tmp_path / "env_rag")
    assert env_settings.rag_mode == "adaptive"

    app = create_app(InSightEngine(_make_settings(tmp_path)))
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["rag_mode"] == "hybrid"
    assert health["available_rag_modes"] == ["hybrid", "adaptive"]

    # UI must expose exactly 2 options inside #rag-mode-select.
    ui_html = client.get("/").text
    select_match = re.search(r'<select[^>]*id="rag-mode-select"[^>]*>(.*?)</select>', ui_html, re.S)
    assert select_match is not None
    options = re.findall(r'<option\s+value="([^"]+)"[^>]*>([^<]+)</option>', select_match.group(1))
    assert options == [("hybrid", "HybridRAG"), ("adaptive", "AdaptiveRAG")]

    # Upload documents to test both HybridRAG and AdaptiveRAG end-to-end.
    assert client.post(
        "/documents/upload",
        files={
            "file": (
                "hybrid_guide.md",
                b"# HybridRAG\n\nQdrant adalah penyimpanan vektor lokal untuk HybridRAG dengan bobot dense 0.7 dan BM25 sparse 0.3.",
                "text/markdown",
            )
        },
    ).status_code == 200
    assert client.post(
        "/documents/upload",
        files={
            "file": (
                "adaptive_guide.md",
                b"# AdaptiveRAG\n\nAdaptiveRAG menganalisis query untuk memilih strategi direct_llm, simple_vector, hybrid_search, atau comprehensive.",
                "text/markdown",
            )
        },
    ).status_code == 200

    # 1. HybridRAG request
    res_hybrid = client.post(
        "/chat",
        json={"question": "Bagaimana bobot dense pada HybridRAG?", "rag_mode": "hybrid"},
    )
    assert res_hybrid.status_code == 200
    body_h = res_hybrid.json()
    assert body_h["rag_mode"] == "hybrid"
    assert body_h["retrieval_strategy"] in {"hybrid_search", "hybrid_search_rewritten"}
    assert body_h["query_analysis"] is None
    assert "0.7" in body_h["answer"]

    # 2. AdaptiveRAG short factual -> direct_llm (or comprehensive if escalated)
    res_adapt_short = client.post(
        "/chat",
        json={"question": "Apa itu Qdrant?", "rag_mode": "adaptive"},
    )
    assert res_adapt_short.status_code == 200
    body_as = res_adapt_short.json()
    assert body_as["rag_mode"] == "adaptive"
    assert body_as["retrieval_strategy"] == "direct_llm"
    assert body_as["query_analysis"]["complexity"] == "simple"
    assert len(body_as["sources"]) <= 2

    # 3. AdaptiveRAG comparative -> comprehensive
    res_adapt_comp = client.post(
        "/chat",
        json={
            "question": "Compare HybridRAG vs AdaptiveRAG strategies across the uploaded documents",
            "rag_mode": "adaptive",
        },
    )
    assert res_adapt_comp.status_code == 200
    body_ac = res_adapt_comp.json()
    assert body_ac["rag_mode"] == "adaptive"
    assert body_ac["retrieval_strategy"] == "comprehensive"
    assert body_ac["query_analysis"]["needs_multiple_sources"] is True

    # 4. Unsupported RAG mode must fail with HTTP 400 without polluting history.
    hist_before = len(client.get("/history").json()["messages"])
    bad_mode = client.post(
        "/chat",
        json={"question": "Test graph rag", "rag_mode": "graph_rag"},
    )
    assert bad_mode.status_code == 400
    assert "Hanya tersedia 2 opsi" in bad_mode.json()["detail"]
    hist_after = len(client.get("/history").json()["messages"])
    assert hist_after == hist_before


def test_resources_dir_env_routing_and_qwen_05b_default(tmp_path: Path, monkeypatch) -> None:
    import os
    from insight.config import configure_resource_env

    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )

    res_root = tmp_path / "resources"
    for key in (
        "INSIGHT_RESOURCES_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "FASTEMBED_CACHE_PATH",
        "TORCH_HOME",
        "OLLAMA_MODELS",
        "XDG_CACHE_HOME",
        "INSIGHT_LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("INSIGHT_RESOURCES_DIR", str(res_root))
    configured = configure_resource_env()
    assert configured == res_root.resolve()
    assert os.environ["HF_HOME"] == str(res_root.resolve() / "huggingface")
    assert os.environ["FASTEMBED_CACHE_PATH"] == str(res_root.resolve() / "fastembed")
    assert os.environ["OLLAMA_MODELS"] == str(res_root.resolve() / "ollama")
    assert os.environ["TORCH_HOME"] == str(res_root.resolve() / "torch")

    env_settings = Settings.from_env(base_dir=tmp_path / "data")
    assert env_settings.llm_model == "unsloth/Qwen3-1.7B-GGUF"
    assert env_settings.resources_dir == res_root.resolve()

    dummy_model = res_root / "ollama" / "qwen05b.bin"
    dummy_model.write_bytes(b"0" * 1024)

    app = create_app(InSightEngine(env_settings))
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["resources_dir"] == str(res_root.resolve())
    assert health["resources_files"] == 1

    res_info = client.get("/resources")
    assert res_info.status_code == 200
    body = res_info.json()
    assert body["file_count"] == 1
    assert body["categories"]["ollama"]["file_count"] == 1
    assert body["items"][0]["path"] == "ollama/qwen05b.bin"


def test_auto_runner_selects_local_gguf_when_ollama_unavailable(tmp_path: Path, monkeypatch) -> None:
    from insight.llm import AutoRunner, LlamaCppRunner, resolve_gguf_model

    res_root = tmp_path / "resources"
    gguf_dir = res_root / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_file = gguf_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    gguf_file.write_bytes(b"gguf-stub")

    settings = Settings(
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "data" / "insight.sqlite",
        qdrant_path=tmp_path / "data" / "qdrant_db",
        upload_dir=tmp_path / "data" / "uploads",
        llm_backend="auto",
        llm_model="qwen2.5:0.5b",
        precision_mode="4bit",
        max_seq_length=2048,
        max_new_tokens=256,
        ollama_url="http://127.0.0.1:19999",
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
        resources_dir=res_root,
    )

    assert resolve_gguf_model(settings) == gguf_file

    captured: dict[str, str] = {}

    def _fake_llamacpp_init(self, model_path: Path, cfg: Settings) -> None:
        captured["model_path"] = str(model_path)
        self.settings = cfg
        self.model_path = model_path
        self.label = f"llamacpp:{model_path.name}"

    monkeypatch.setattr(LlamaCppRunner, "__init__", _fake_llamacpp_init)
    monkeypatch.setattr(LlamaCppRunner, "generate", lambda self, q, ctx: f"GGUF-OK: {q}")
    monkeypatch.setattr("insight.llm.resolve_ollama_model", lambda _cfg, timeout=0.5: None)

    runner = AutoRunner(settings)
    assert runner.label == "llamacpp:qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert captured["model_path"] == str(gguf_file)
    assert runner.generate("Apa isi dokumen?", []) == "GGUF-OK: Apa isi dokumen?"


def test_unsloth_qwen3_gguf_resolution_priority_and_think_stripping(tmp_path: Path, monkeypatch) -> None:
    from insight.llm import AutoRunner, LlamaCppRunner, _strip_thinking_tags, resolve_gguf_model
    from insight.vectorstore import RetrievedChunk

    res_root = tmp_path / "resources"
    gguf_dir = res_root / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    other_gguf = gguf_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    other_gguf.write_bytes(b"old")

    settings = Settings(
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "data" / "insight.sqlite",
        qdrant_path=tmp_path / "data" / "qdrant_db",
        upload_dir=tmp_path / "data" / "uploads",
        llm_backend="auto",
        llm_model="unsloth/Qwen3-1.7B-GGUF",
        precision_mode="4bit",
        max_seq_length=2048,
        max_new_tokens=256,
        ollama_url="http://127.0.0.1:19999",
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
        resources_dir=res_root,
    )

    # Must NOT pick unrelated qwen2.5 GGUF when unsloth/Qwen3-1.7B-GGUF is requested.
    assert resolve_gguf_model(settings) is None

    qwen3_gguf = gguf_dir / "Qwen3-1.7B-UD-Q4_K_XL.gguf"
    qwen3_gguf.write_bytes(b"qwen3")
    assert resolve_gguf_model(settings) == qwen3_gguf

    # Even if Ollama has an unrelated model, AutoRunner must prefer the matching local GGUF.
    monkeypatch.setattr("insight.llm.resolve_ollama_model", lambda _cfg, timeout=0.5: "other-ollama:latest")

    captured_calls: list[dict] = []

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            captured_calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<think>\ninternal reasoning\n</think>\n\nJawaban final dari dokumen."
                        }
                    }
                ]
            }

    def _fake_init(self, model_path: Path, cfg: Settings) -> None:
        self._model = FakeLlama()
        self.settings = cfg
        self.model_path = model_path
        self._is_qwen3 = True
        self.label = f"llamacpp:{model_path.name}"

    monkeypatch.setattr(LlamaCppRunner, "__init__", _fake_init)

    runner = AutoRunner(settings)
    assert runner.label == "llamacpp:Qwen3-1.7B-UD-Q4_K_XL.gguf"
    chunk = RetrievedChunk(doc_id="d1", filename="doc.md", section="Intro", page_label="1", chunk_index=0, text="Isi doc.", score=0.9)
    answer = runner.generate("Apa isinya?", [chunk])
    assert answer == "Jawaban final dari dokumen."
    assert captured_calls[0]["messages"][-1]["content"].endswith("/no_think")
    assert captured_calls[0]["temperature"] == 0.7
    assert captured_calls[0]["top_p"] == 0.8
    assert _strip_thinking_tags("<think>unclosed") == ""


def test_qdrant_migrates_when_embedder_name_changes_same_dimension(tmp_path: Path, monkeypatch) -> None:
    from insight.parser import TextChunk
    from insight.vectorstore import HashingEmbedder, LocalVectorStore

    class _NamedEmbedderV1(HashingEmbedder):
        name = "embedder-v1-384d"

    class _NamedEmbedderV2(HashingEmbedder):
        name = "embedder-v2-384d"

    q_path = tmp_path / "qdrant_db"
    monkeypatch.setattr("insight.vectorstore.build_embedder", lambda _name: _NamedEmbedderV1())
    store1 = LocalVectorStore(qdrant_path=q_path, embed_model="v1")
    store1.upsert_chunks(
        doc_id="doc1",
        filename="paper.pdf",
        chunks=[
            TextChunk(
                chunk_index=0,
                text="Disusun oleh Budi Santoso dan Siti Aminah.",
                section="Halaman 1",
                page_label="1",
            )
        ],
    )
    assert store1.migrated_points == 0
    store1.close()

    monkeypatch.setattr("insight.vectorstore.build_embedder", lambda _name: _NamedEmbedderV2())
    store2 = LocalVectorStore(qdrant_path=q_path, embed_model="v2")
    assert store2.migrated_points == 1
    hits = store2.search("siapa penyusun paper", top_k_candidates=4, top_k_final=2)
    assert hits and "Budi Santoso" in hits[0].text
    store2.close()


def test_conversation_memory_front_matter_and_toc_penalty(tmp_path: Path, monkeypatch) -> None:
    from insight.vectorstore import _toc_noise_ratio

    monkeypatch.setattr(
        "insight.vectorstore.build_embedder",
        lambda _name: __import__("insight.vectorstore", fromlist=["HashingEmbedder"]).HashingEmbedder(),
    )
    toc_sample = (
        "DAFTAR ISI ......................................................... i\n"
        "BAB I PENDAHULUAN ................................................. 1\n"
        "BAB II METODOLOGI .................................................. 4\n"
        "ABSTRAK ............................................................ vii"
    )
    assert _toc_noise_ratio(toc_sample) >= 0.7

    doc_path = tmp_path / "laporan.md"
    doc_path.write_text(
        "# Halaman 1\n\nMINI PROJECT KELOMPOK 9\nDisusun oleh: Andi Pratama, Dewi Lestari\n\n"
        "# Daftar Isi\n\n"
        + toc_sample
        + "\n\n# Metodologi\n\nProyek ini membandingkan Random Forest dan XGBoost pada data pertanian.",
        encoding="utf-8",
    )

    engine = InSightEngine(_make_settings(tmp_path))
    captured_histories: list[list[dict[str, str]] | None] = []

    def _capture_generate(question: str, contexts, history=None) -> str:
        captured_histories.append(history)
        return " | ".join(c.text[:40] for c in contexts)

    monkeypatch.setattr(engine.llm, "generate", _capture_generate)
    engine.ingest_file(doc_path)

    res1 = engine.ask("pada bagian cover siapa penulisnya?", session_id="s-mem", rag_mode="hybrid")
    assert any("Andi Pratama" in s["excerpt"] for s in res1.sources)
    assert captured_histories[-1] == []

    res2 = engine.ask("siapa saja?", session_id="s-mem", rag_mode="hybrid")
    assert any("Andi Pratama" in s["excerpt"] for s in res2.sources)
    assert captured_histories[-1] is not None
    assert len(captured_histories[-1]) >= 2
    assert captured_histories[-1][0]["content"] == "pada bagian cover siapa penulisnya?"
    engine.close()


def test_true_corpus_wide_bm25_hybrid_retrieval(tmp_path: Path) -> None:
    from insight.parser import TextChunk
    from insight.vectorstore import LocalVectorStore

    class BiasDenseEmbedder:
        dimension = 3
        name = "bias-dense-3d"

        def embed(self, texts: list[str]) -> list[list[float]]:
            vecs: list[list[float]] = []
            for t in texts:
                if "PASAL_KHUSUS_99" in t and "QUERY:" not in t:
                    # Orthogonal to query vector so dense-only top_k_candidates=4 misses it completely.
                    vecs.append([0.0, 1.0, 0.0])
                elif "QUERY:" in t:
                    vecs.append([1.0, 0.0, 0.0])
                else:
                    # Moderate dense similarity (~0.30) for filler chunks.
                    vecs.append([0.3, 0.9539, 0.0])
            return vecs

    store = LocalVectorStore(qdrant_path=tmp_path / "qdrant", embed_model="hashing-fallback")
    store.embedder = BiasDenseEmbedder()
    store._ensure_collection()

    chunks = [
        TextChunk(chunk_index=i, text=f"Dokumen umum paragraf pengisi nomor {i}.", section="Umum", page_label="1")
        for i in range(15)
    ]
    chunks.append(
        TextChunk(
            chunk_index=15,
            text="Ketentuan PASAL_KHUSUS_99 mengatur sanksi administratif lokal.",
            section="Sanksi",
            page_label="16",
        )
    )
    store.upsert_chunks("doc-hybrid", "regulasi.md", chunks)

    hits = store.search(
        query="QUERY: PASAL_KHUSUS_99",
        top_k_candidates=4,
        top_k_final=2,
        strategy="hybrid_search",
        dense_weight=0.70,
        sparse_weight=0.30,
    )
    assert any("PASAL_KHUSUS_99" in h.text for h in hits)
    store.close()


def test_docling_memory_release_and_chat_stream_sse(tmp_path: Path, monkeypatch) -> None:
    import json
    import insight.parser as parser_mod
    from insight.llm import _stream_filter_think

    # 1. Verify _stream_filter_think strips split <think>...</think> blocks cleanly.
    raw_chunks = iter(["<thi", "nk>rahasia", " internal</thi", "nk>\n\nJawaban ", "bersih."])
    assert "".join(_stream_filter_think(raw_chunks)) == "Jawaban bersih."

    # 2. Verify Docling converter is released after parse_file unless INSIGHT_KEEP_DOCLING_LOADED=true.
    monkeypatch.delenv("INSIGHT_KEEP_DOCLING_LOADED", raising=False)
    parser_mod._docling_converter = object()
    monkeypatch.setattr(parser_mod, "_get_docling_converter", lambda: type("C", (), {"convert": lambda self, _p: type("R", (), {"document": type("D", (), {"export_to_markdown": lambda self, **kw: "Isi"})()})()})())
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")
    parsed = parser_mod.parse_file(pdf_file)
    assert parsed.parser_used == "docling"
    assert parser_mod._docling_converter is None

    # 3. Verify POST /chat/stream emits meta, token, and done SSE events and saves history.
    engine = InSightEngine(_make_settings(tmp_path))
    doc = tmp_path / "catatan.md"
    doc.write_text("# Ringkasan\n\nInSight mendukung streaming token SSE.", encoding="utf-8")
    engine.ingest_file(doc)
    monkeypatch.setattr(
        engine.llm,
        "generate_stream",
        lambda question, contexts, history=None: iter(["Halo ", "dunia!"]),
    )

    client = TestClient(create_app(engine=engine))
    with client.stream(
        "POST",
        "/chat/stream",
        json={"question": "Apa yang didukung InSight?", "session_id": "s-stream", "rag_mode": "hybrid"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    events = []
    for frame in body.strip().split("\n\n"):
        data_line = next((ln for ln in frame.splitlines() if ln.startswith("data:")), None)
        if data_line:
            events.append(json.loads(data_line[5:].strip()))

    assert any(e.get("type") == "meta" and e.get("rag_mode") == "hybrid" for e in events)
    assert "".join(e.get("delta", "") for e in events if e.get("type") == "token") == "Halo dunia!"
    assert any(e.get("type") == "done" and e.get("answer") == "Halo dunia!" for e in events)

    hist = client.get("/history", params={"session_id": "s-stream"}).json()["messages"]
    assert hist[-1]["role"] == "assistant"
    assert hist[-1]["content"] == "Halo dunia!"
    engine.close()
