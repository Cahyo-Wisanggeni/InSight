"""Lightweight document parsing (PDF/Office/HTML/Text) and structure-aware Markdown chunking."""

from __future__ import annotations

import re
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

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".xhtml",
}

PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*INSIGHT_PAGE:(\d+)\s*-->\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")


@dataclass(frozen=True)
class ParsedDocument:
    markdown: str
    parser_used: str


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    section: str
    page_label: str = ""


@dataclass
class _MarkdownBlock:
    text: str
    pages: list[int]
    preceded_by_page_break: bool = False


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

    if ext in DOCUMENT_EXTENSIONS:
        try:
            parsed = _parse_document_file(file_path, ext)
            if parsed is not None:
                return parsed
        except Exception as exc:
            raise RuntimeError(f"Gagal memproses '{file_path.name}': {exc}") from exc

    raise ValueError(f"Unsupported file type: '{ext or 'unknown'}'")


def _parse_document_file(file_path: Path, ext: str) -> ParsedDocument | None:
    """Extract Markdown from PDF, Office, and HTML files using lightweight Python parsers."""
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
                    pages.append(f"<!-- INSIGHT_PAGE:{idx + 1} -->\n\n{text}")
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
                for r_idx, row in enumerate(sheet.iter_rows(values_only=True)):
                    vals = ["" if v is None else str(v).strip() for v in row]
                    if any(vals):
                        rows_out.append("| " + " | ".join(vals) + " |")
                        if len(rows_out) == 1:
                            rows_out.append("| " + " | ".join("---" for _ in vals) + " |")
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


def _split_table_cells(line: str) -> list[str] | None:
    """Split a Markdown table row into stripped cell values, or return None if not a table row."""
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells if len(cells) >= 2 else None


def _analyze_table_block(text: str) -> tuple[list[str], int, bool] | None:
    """Analyze a Markdown block to determine whether it is a pipe table.

    Returns (lines, column_count, has_separator_row) or None if the block is not a table.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    col_count: int | None = None
    for line in lines:
        cells = _split_table_cells(line)
        if cells is None:
            return None
        if col_count is None:
            col_count = len(cells)
        elif len(cells) != col_count:
            return None

    assert col_count is not None
    has_sep = len(lines) >= 2 and bool(TABLE_SEP_RE.match(lines[1]))
    return lines, col_count, has_sep


def _extract_blocks(markdown: str) -> list[_MarkdownBlock]:
    """Split Markdown into paragraph/table blocks while tracking page boundary markers."""
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[_MarkdownBlock] = []
    current_page: int | None = None
    saw_page_break = False
    pending_lines: list[str] = []

    def flush_pending() -> None:
        nonlocal saw_page_break
        if not pending_lines:
            return
        content = "\n".join(pending_lines).strip()
        pending_lines.clear()
        if not content:
            return
        pages = [current_page] if current_page is not None else []
        blocks.append(
            _MarkdownBlock(
                text=content,
                pages=pages,
                preceded_by_page_break=saw_page_break,
            )
        )
        saw_page_break = False

    for raw_line in normalized.split("\n"):
        marker_match = PAGE_MARKER_RE.match(raw_line)
        if marker_match:
            flush_pending()
            current_page = int(marker_match.group(1))
            saw_page_break = True
            continue
        if not raw_line.strip():
            flush_pending()
            continue
        pending_lines.append(raw_line)

    flush_pending()
    return blocks


def _merge_split_tables(blocks: list[_MarkdownBlock]) -> list[_MarkdownBlock]:
    """Merge consecutive Markdown table blocks separated only by a PDF page boundary."""
    if len(blocks) < 2:
        return blocks

    merged: list[_MarkdownBlock] = []
    for block in blocks:
        if not merged or not block.preceded_by_page_break:
            merged.append(block)
            continue

        prev = merged[-1]
        prev_info = _analyze_table_block(prev.text)
        curr_info = _analyze_table_block(block.text)
        if prev_info is None or curr_info is None:
            merged.append(block)
            continue

        prev_lines, prev_cols, prev_has_sep = prev_info
        curr_lines, curr_cols, curr_has_sep = curr_info

        # Require the preceding table to be a real Markdown table (header + separator)
        # and the continuation block to have the exact same column count.
        if not prev_has_sep or prev_cols != curr_cols:
            merged.append(block)
            continue

        continuation_rows: list[str]
        if curr_has_sep:
            prev_header = [c.lower() for c in (_split_table_cells(prev_lines[0]) or [])]
            curr_header = [c.lower() for c in (_split_table_cells(curr_lines[0]) or [])]
            is_repeated_header = curr_header == prev_header or all(c == "" for c in curr_header)
            if not is_repeated_header:
                # Different table headers on the next page -> treat as a separate table.
                merged.append(block)
                continue
            continuation_rows = curr_lines[2:]
        else:
            continuation_rows = curr_lines

        if continuation_rows:
            prev.text = "\n".join(prev_lines + continuation_rows)
        for p in block.pages:
            if p not in prev.pages:
                prev.pages.append(p)
        prev.pages.sort()

    return merged


def _format_page_label(pages: list[int]) -> str:
    if not pages:
        return ""
    lo, hi = min(pages), max(pages)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _split_large_table(table_text: str, chunk_size: int) -> list[str]:
    """Split an oversized Markdown table by rows while repeating header + separator."""
    info = _analyze_table_block(table_text)
    if info is None:
        return [table_text]
    lines, _cols, has_sep = info
    if not has_sep or len(lines) <= 2:
        return [table_text]

    header_block = "\n".join(lines[:2])
    body_rows = lines[2:]
    pieces: list[str] = []
    cur_rows: list[str] = []

    for row in body_rows:
        candidate = header_block + "\n" + "\n".join(cur_rows + [row])
        if cur_rows and len(candidate) > chunk_size:
            pieces.append(header_block + "\n" + "\n".join(cur_rows))
            cur_rows = [row]
        else:
            cur_rows.append(row)

    if cur_rows:
        pieces.append(header_block + "\n" + "\n".join(cur_rows))
    return pieces or [table_text]


def chunk_markdown(
    markdown: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Split Markdown into overlapping chunks while preserving heading and page context."""
    text = markdown.strip()
    if not text:
        return []

    blocks = _merge_split_tables(_extract_blocks(text))
    if not blocks:
        return []

    chunks: list[TextChunk] = []
    current_section = "General"
    buffer_section = current_section
    buffer = ""
    buffer_pages: list[int] = []

    def flush(content: str, section: str, pages: list[int]) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                text=cleaned,
                section=section,
                page_label=_format_page_label(pages),
            )
        )

    def _merge_pages(existing: list[int], incoming: list[int]) -> list[int]:
        combined = sorted(set(existing) | set(incoming))
        return combined

    for block in blocks:
        para = block.text
        if para.startswith("#"):
            if buffer:
                flush(buffer, buffer_section, buffer_pages)
                buffer = ""
                buffer_pages = []
            first_line = para.splitlines()[0].lstrip("#").strip()
            if first_line:
                current_section = first_line[:120]
            buffer_section = current_section

        if len(para) > chunk_size:
            if buffer:
                flush(buffer, buffer_section, buffer_pages)
                buffer = ""
                buffer_pages = []

            table_info = _analyze_table_block(para)
            if table_info is not None and table_info[2]:
                for piece in _split_large_table(para, chunk_size):
                    flush(piece, current_section, block.pages)
            else:
                step = max(1, chunk_size - chunk_overlap)
                for start in range(0, len(para), step):
                    flush(para[start : start + chunk_size], current_section, block.pages)
            buffer_section = current_section
            continue

        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= chunk_size:
            if not buffer:
                buffer_section = current_section
            buffer = candidate
            buffer_pages = _merge_pages(buffer_pages, block.pages)
        else:
            prev_last_page = [buffer_pages[-1]] if buffer_pages else []
            flush(buffer, buffer_section, buffer_pages)
            overlap_tail = buffer[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            if overlap_tail:
                buffer = f"{overlap_tail}\n\n{para}".strip()
                buffer_pages = _merge_pages(prev_last_page, block.pages)
            else:
                buffer = para
                buffer_pages = list(block.pages)
            buffer_section = current_section

    if buffer:
        flush(buffer, buffer_section, buffer_pages)

    return chunks
