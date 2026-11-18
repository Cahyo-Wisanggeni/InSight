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

            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            markdown = result.document.export_to_markdown().strip()
            return ParsedDocument(markdown=markdown, parser_used="docling")
        except Exception as exc:
            fallback = _fallback_parse(file_path, ext)
            if fallback is not None:
                return fallback
            if isinstance(exc, ImportError):
                raise RuntimeError(
                    f"Format '{ext}' membutuhkan Docling. Install dengan: pip install docling"
                ) from exc
            raise RuntimeError(f"Gagal memproses '{file_path.name}': {exc}") from exc

    raise ValueError(f"Unsupported file type: '{ext or 'unknown'}'")


def _fallback_parse(file_path: Path, ext: str) -> ParsedDocument | None:
    """Fallback extractors using lightweight Python parsers when Docling is unavailable."""
    if ext == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(file_path))
        pages: list[str] = []
        try:
            for idx in range(len(pdf)):
                page = pdf[idx]
                textpage = page.get_textpage()
                text = textpage.get_text_range().strip()
                textpage.close()
                page.close()
                if text:
                    pages.append(f"## Halaman {idx + 1}\n\n{text}")
        finally:
            pdf.close()
        return ParsedDocument(markdown="\n\n".join(pages).strip(), parser_used="pypdfium2")

    if ext == ".docx":
        import docx

        document = docx.Document(str(file_path))
        blocks: list[str] = []
        for para in document.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style_name = getattr(para.style, "name", "") or ""
            if style_name.startswith("Heading"):
                level_str = style_name.replace("Heading", "").strip()
                level = int(level_str) if level_str.isdigit() else 2
                level = max(1, min(level, 6))
                blocks.append(f"{'#' * level} {text}")
            else:
                blocks.append(text)
        for table in document.tables:
            rows = []
            for r_idx, row in enumerate(table.rows):
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    rows.append("| " + " | ".join("---" for _ in cells) + " |")
            if rows:
                blocks.append("\n".join(rows))
        return ParsedDocument(markdown="\n\n".join(blocks).strip(), parser_used="python-docx")

    if ext == ".pptx":
        from pptx import Presentation

        prs = Presentation(str(file_path))
        slides_md: list[str] = []
        for idx, slide in enumerate(prs.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    txt = shape.text_frame.text.strip()
                    if txt:
                        parts.append(txt)
            if parts:
                slides_md.append(f"## Slide {idx}\n\n" + "\n\n".join(parts))
        return ParsedDocument(markdown="\n\n".join(slides_md).strip(), parser_used="python-pptx")

    if ext == ".xlsx":
        import openpyxl

        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        sheets_md: list[str] = []
        try:
            for sheet in wb.worksheets:
                rows_out: list[str] = []
                for row in sheet.iter_rows(values_only=True):
                    vals = ["" if v is None else str(v).strip() for v in row]
                    if any(vals):
                        rows_out.append("| " + " | ".join(vals) + " |")
                if rows_out:
                    sheets_md.append(f"## Sheet: {sheet.title}\n\n" + "\n".join(rows_out))
        finally:
            wb.close()
        return ParsedDocument(markdown="\n\n".join(sheets_md).strip(), parser_used="openpyxl")

    if ext in {".html", ".htm", ".xhtml"}:
        from bs4 import BeautifulSoup

        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
        return ParsedDocument(markdown="\n\n".join(lines).strip(), parser_used="beautifulsoup4")

    return None


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
