"""Document parsing (Docling-first) and structure-aware Markdown chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".py",
    ".tex",
    ".yaml",
    ".yml",
}

DOCLING_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".xhtml",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".gif",
    ".asciidoc",
    ".adoc",
}


@dataclass(frozen=True)
class ParsedDocument:
    markdown: str
    parser_used: str


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    section: str


def parse_file(file_path: Path) -> ParsedDocument:
    """Convert an uploaded file into clean Markdown."""
    ext = file_path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        raw = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if ext == ".json":
            markdown = f"```json\n{raw}\n```"
        elif ext == ".py":
            markdown = f"```python\n{raw}\n```"
        else:
            markdown = raw
        return ParsedDocument(markdown=markdown, parser_used="native_text")

    if ext in DOCLING_EXTENSIONS:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                f"Format '{ext}' requires Docling. Install it with: pip install docling"
            ) from exc

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        markdown = result.document.export_to_markdown().strip()
        return ParsedDocument(markdown=markdown, parser_used="docling")

    raise ValueError(f"Unsupported file type: '{ext or 'unknown'}'")


def chunk_markdown(
    markdown: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Split Markdown into overlapping chunks while preserving heading context."""
    text = markdown.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[TextChunk] = []
    current_section = "General"
    buffer_section = current_section
    buffer = ""

    def flush(content: str, section: str) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                text=cleaned,
                section=section,
            )
        )

    for para in paragraphs:
        if para.startswith("#"):
            if buffer:
                flush(buffer, buffer_section)
                buffer = ""
            first_line = para.splitlines()[0].lstrip("#").strip()
            if first_line:
                current_section = first_line[:120]
            buffer_section = current_section

        if len(para) > chunk_size:
            if buffer:
                flush(buffer, buffer_section)
                buffer = ""
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(para), step):
                flush(para[start : start + chunk_size], current_section)
            buffer_section = current_section
            continue

        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= chunk_size:
            if not buffer:
                buffer_section = current_section
            buffer = candidate
        else:
            flush(buffer, buffer_section)
            overlap_tail = buffer[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            buffer = f"{overlap_tail}\n\n{para}".strip() if overlap_tail else para
            buffer_section = current_section

    if buffer:
        flush(buffer, buffer_section)

    return chunks
