<div align="center">

  <h1><sub><picture><source media="(prefers-color-scheme: dark)" srcset="insight/static/logo-dark.svg" /><source media="(prefers-color-scheme: light)" srcset="insight/static/logo.svg" /><img src="insight/static/logo-dark.svg" alt="InSight Logo" width="34" height="34" /></picture></sub> InSight</h1>

  <p>
    <strong>Workspace <em>Retrieval-Augmented Generation</em> (RAG) Dokumen Privat 100% Lokal &amp; Offline</strong>
  </p>

  <p>
    Ajukan pertanyaan langsung ke koleksi dokumen Anda (PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, CSV, JSON, YAML, LaTeX, dan Python) secara privat di perangkat sendiri menggunakan parser ringan, Qdrant, SQLite, dan LLM lokal tanpa mengirim data ke API pihak ketiga.
  </p>

  <p>
    <a href="https://github.com/Cahyo-Wisanggeni/InSight/releases/tag/v0.1.0"><img src="https://img.shields.io/badge/Release-v0.1.0-2563EB?style=flat-square" alt="Release v0.1.0"/></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
    <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/VectorDB-Qdrant-DC2626?style=flat-square" alt="Qdrant"/></a>
    <a href="https://www.sqlite.org/"><img src="https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite"/></a>
    <a href="https://github.com/ggml-org/llama.cpp"><img src="https://img.shields.io/badge/LLM-llama.cpp_%2F_Ollama-7C3AED?style=flat-square" alt="llama.cpp / Ollama"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Lisensi-MIT-EAB308?style=flat-square" alt="MIT License"/></a>
  </p>

  <p>
    <a href="#pratinjau-aplikasi">Pratinjau</a> &middot;
    <a href="#keunggulan--fitur-utama">Fitur Utama</a> &middot;
    <a href="#rincian-mode-rag--pilihan-model">Mode RAG &amp; Model</a> &middot;
    <a href="#panduan-memulai-cepat">Panduan Memulai</a> &middot;
    <a href="#ringkasan-api">API</a>
  </p>

</div>

<br/>

<div align="center">
  <a href="https://github.com/Cahyo-Wisanggeni/InSight/releases/download/v0.1.0/InSight-Demo.mp4">
    <img src="assets/demo.webp" alt="Demo InSight Workspace" width="950"/>
  </a>
  <p>
    <sub><a href="https://github.com/Cahyo-Wisanggeni/InSight/releases/download/v0.1.0/InSight-Demo.mp4">video demo lengkap</a></sub>
  </p>
</div>

---

## Daftar Isi

- [Keunggulan & Fitur Utama](#keunggulan--fitur-utama)
- [Pratinjau Aplikasi](#pratinjau-aplikasi)
- [Rincian Mode RAG & Pilihan Model](#rincian-mode-rag--pilihan-model)
- [Panduan Memulai Cepat](#panduan-memulai-cepat)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Ringkasan API](#ringkasan-api)
- [Struktur Repositori](#struktur-repositori)
- [Pengujian](#pengujian)
- [Lisensi](#lisensi)

---

## Keunggulan & Fitur Utama

**Privasi 100% Lokal & Offline**<br>
FastAPI, parser dokumen ringan, Qdrant, SQLite, direktori `resources/`, dan LLM lokal (`llama.cpp` GGUF / Ollama / `local_smart`) berjalan sepenuhnya di PC Anda. Dokumen dan pertanyaan tidak pernah keluar dari perangkat.

**Pilihan Model LLM Dinamis dari UI**<br>
Model LLM menggunakan Qwen antara lain: `Qwen3-0.6B`, `Qwen3.5-0.8B`, `Qwen3-1.7B`, atau model Ollama lokal tanpa perlu restart server.

**Parsing Ringan & Sitasi Halaman Akurat**<br>
Mengonversi PDF, DOCX, PPTX, XLSX, dan HTML ke Markdown terstruktur tanpa beban PyTorch/OCR, melacak nomor halaman per chunk, serta menggabungkan tabel Markdown yang terpecah lintas halaman.

**Folder Proyek & Pencarian Terfokus**<br>
Kelompokkan dokumen ke dalam folder proyek, lalu arahkan pencarian ke proyek tertentu atau seluruh dokumen terindeks langsung dari kolom *composer*.

**Streaming Jawaban Real-Time & Tema Adaptif**<br>
Mendukung *token streaming* via Server-Sent Events, mode gelap/terang, riwayat percakapan SQLite, serta penghapusan vektor Qdrant otomatis saat dokumen dihapus.

**Folder `resources/` Terpusat & Proteksi Opsional**<br>
Seluruh cache model terpusat di `./resources` yang dapat dipantau via UI, ditambah proteksi kata sandi opsional dengan *rate-limiting* dan TTL token.

---

## Pratinjau Aplikasi

Antarmuka Web UI minimalis dengan dukungan **Mode Gelap** dan **Mode Terang**, manajemen folder proyek, pemilih model & mode RAG di toolbar atas, serta pengalaman tanya-jawab dokumen yang bersih:

| Mode Gelap | Mode Terang |
| :---: | :---: |
| <img src="assets/ui_preview_dark.png" alt="Tampilan Mode Gelap InSight" width="470"/> | <img src="assets/ui_preview_light.png" alt="Tampilan Mode Terang InSight" width="470"/> |

---

## Rincian Mode RAG & Pilihan Model

1. **HybridRAG (`hybrid`)**:
   - Menggabungkan pencarian vektor *dense* dan *sparse* BM25/leksikal (`0.70 dense + 0.30 sparse`) dengan evaluasi relevansi dan *query rewriting* otomatis 1x bila hasil awal kurang relevan.
2. **AdaptiveRAG (`adaptive`)**:
   - Menggunakan `QueryAnalyzer` untuk menilai kompleksitas (`simple`, `medium`, `complex`), faktualitas, dan kebutuhan multi-sumber, lalu merutekan otomatis ke `direct_llm`, `simple_vector`, `hybrid_search` (`0.8/0.2` atau `0.6/0.4`), atau `comprehensive` dengan eskalasi otomatis.
3. **Model Selection Dinamis**:
   - Tersedia langsung dari dropdown toolbar UI maupun modal Pengaturan, termasuk opsi presisi (`4bit` / `8bit` / `full`) untuk menyesuaikan spesifikasi RAM/GPU lokal.

---

## Panduan Memulai Cepat

### 1. Prasyarat Sistem
- **Python 3.10+** (teruji pada Python 3.10 hingga 3.13).
- Koneksi internet hanya diperlukan saat mengunduh paket dan file model GGUF pertama kali (setelah itu dapat berjalan 100% offline).

### 2. Kloning Repositori & Instalasi Dependensi

```bash
git clone https://github.com/Cahyo-Wisanggeni/InSight.git
cd InSight

# Membuat dan mengaktifkan virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Pasang seluruh paket dependensi
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi

#### Pada Windows (Sekali Klik):
- Klik dua kali berkas **`InSight.bat`** (atau jalankan `python launch.py`). Server berjalan otomatis di latar belakang, UI langsung terbuka di browser, dan server berhenti bersih saat jendela UI ditutup.

#### Melalui Terminal (Mode Manual):
```bash
python run.py
```

#### Menggunakan Ollama Lokal (Opsional):
Pastikan [Ollama](https://ollama.com/) aktif dan model sudah diunduh (mis. `ollama pull qwen2.5:7b`), lalu jalankan:
```powershell
# Windows PowerShell
$env:INSIGHT_LLM_BACKEND="ollama"
$env:INSIGHT_LLM_MODEL="qwen2.5:7b"
python run.py
```
```bash
# Linux / macOS
INSIGHT_LLM_BACKEND=ollama INSIGHT_LLM_MODEL=qwen2.5:7b python run.py
```

### 4. Buka Aplikasi
Akses `http://127.0.0.1:8000` di browser, unggah dokumen ke folder proyek di panel kiri, pilih model dan mode RAG, lalu ajukan pertanyaan Anda.

---

## Arsitektur Sistem

```mermaid
flowchart TD
    subgraph UI ["Web UI (insight/static/index.html)"]
        A[Folder Proyek, Upload Dokumen & Pengaturan]
        B[Composer Pertanyaan + Pilih Model & Mode RAG]
        C[Tampilan Jawaban Streaming SSE & Sitasi Halaman]
    end

    subgraph API ["Backend FastAPI (run.py & insight/api.py)"]
        D[Lightweight Parsers: pypdfium2, python-docx, python-pptx, openpyxl, bs4]
        E[Chunking Markdown + Penggabungan Tabel Lintas Halaman]
        F[Dual-RAG Router: HybridRAG & AdaptiveRAG]
    end

    subgraph STORE ["Penyimpanan & Inferensi 100% Lokal"]
        G[(Qdrant: Dense + BM25 Sparse)]
        H[(SQLite: Proyek, Dokumen & Riwayat Chat)]
        I[LLM Lokal: llama.cpp GGUF / Ollama / local_smart]
    end

    A -->|Upload| D --> E --> G
    E --> H
    B -->|HTTP / SSE + rag_mode| F
    F <-->|Retrieval & Grading| G
    F -->|Prompt + Konteks Terverifikasi| I
    I -->|Streaming Token| C
    F --> H
```

| Komponen | Spesifikasi |
|---|---|
| **Backend Web & API** | FastAPI + Uvicorn (termasuk SSE streaming di `/chat/stream`) |
| **Parser Dokumen** | `pypdfium2`, `python-docx`, `python-pptx`, `openpyxl`, `beautifulsoup4` |
| **Vector Database** | Qdrant (`Dense` + `BM25Okapi` Sparse) tanpa Docker |
| **Database Metadata** | SQLite (`data/insight.sqlite`) untuk proyek, dokumen, dan riwayat sesi |
| **Strategi RAG** | `HybridRAG` (0.70/0.30 + rewrite) & `AdaptiveRAG` (`QueryAnalyzer` + routing) |
| **Inferensi LLM Lokal** | `llama-cpp-python` (GGUF `Qwen3-0.6B` / `3.5-0.8B` / `3-1.7B`) atau `Ollama` |
| **Keamanan Opsional** | Session token `X-InSight-Token` dengan TTL dan rate-limit login (`HTTP 429`) |

---

## Ringkasan API

FastAPI menyediakan dokumentasi interaktif otomatis (Swagger UI) di `http://127.0.0.1:8000/docs`.

<details>
<summary><b>Klik untuk melihat daftar lengkap endpoint REST &amp; SSE</b></summary>

| Method & Path | Fungsi |
|---|---|
| `GET /health` | Cek status server, backend LLM, metode RAG aktif, status embedder, folder `resources`, dan jumlah dokumen |
| `GET /resources` | Pantau daftar file dan ukuran cache/model terunduh di `INSIGHT_RESOURCES_DIR` |
| `POST /auth/login` | Tukar `INSIGHT_PASSWORD` menjadi token sesi `X-InSight-Token` (rate-limited + TTL) |
| `GET /models` | Ambil daftar model LLM lokal dan preset yang tersedia beserta status aktif/unduhan |
| `POST /models/select` | Beralih model LLM aktif secara instan (`qwen3-0.6b`, `qwen3.5-0.8b`, `qwen3-1.7b`, atau Ollama) |
| `POST /documents/upload` | Unggah dan indeks dokumen ke Qdrant + SQLite |
| `GET /documents` | Ambil daftar dokumen terindeks |
| `DELETE /documents/{doc_id}` | Hapus dokumen dari SQLite dan vektornya dari Qdrant |
| `POST /chat` | Ajukan pertanyaan dengan `rag_mode` (`hybrid` atau `adaptive`) dan simpan ke riwayat SQLite |
| `POST /chat/stream` | Ajukan pertanyaan dengan streaming token Server-Sent Events (SSE) langsung ke UI |
| `GET /history` | Ambil riwayat chat per sesi |
| `DELETE /history` | Bersihkan riwayat chat per sesi |
| `GET /sessions` | Ambil daftar sesi percakapan tersimpan |

</details>

---

## Struktur Repositori

```text
InSight/
├── .env.example
├── .gitignore
├── InSight.bat
├── LICENSE
├── README.md
├── launch.py
├── pyproject.toml
├── requirements.txt
├── run.py
├── assets/
│   ├── demo.webp
│   ├── ui_preview_dark.png
│   └── ui_preview_light.png
├── insight/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── llm.py
│   ├── parser.py
│   ├── rag.py
│   ├── storage.py
│   ├── vectorstore.py
│   └── static/
│       ├── index.html
│       ├── logo.svg
│       └── logo-dark.svg
└── tests/
    └── test_insight.py
```

---

## Pengujian

Jalankan test suite dengan:

```bash
pytest -v
```

---

## Lisensi

Didistribusikan di bawah lisensi **[MIT License](LICENSE)**.
