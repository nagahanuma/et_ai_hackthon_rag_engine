"""
Multi-format streaming document parser.
Supports PDF (PyMuPDF with Markdown table detection), TXT, and Markdown files.
Uses streaming generators to maintain flat O(1) memory consumption during ingestion.
"""

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, List, Dict, Any, Optional

# pyrefly: ignore [missing-import]
import pymupdf

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedBlock:
    """Represents a text paragraph or a markdown table extracted from a page."""
    block_type: str  # 'text' or 'table'
    content: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocumentMetadata:
    """High-level summary of a document before chunking."""
    filename: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    file_type: str


class DocumentParser:
    """Streaming document parser designed for large-scale enterprise ingestion."""

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """
        Streams file in 64KB blocks to compute SHA-256 hash without high memory usage.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @classmethod
    def get_document_info(cls, file_path: Path) -> ParsedDocumentMetadata:
        """Quickly inspects document headers and metadata before processing."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = cls.compute_sha256(path)
        file_size = path.stat().st_size
        file_ext = path.suffix.lower().lstrip(".")

        total_pages = 1
        if file_ext == "pdf":
            doc = pymupdf.open(str(path))
            total_pages = len(doc)
            doc.close()

        logger.info(f"Inspected '{path.name}': {total_pages} pages, {file_size} bytes, type: {file_ext}")
        return ParsedDocumentMetadata(
            filename=path.name,
            file_hash=file_hash,
            file_size_bytes=file_size,
            total_pages=total_pages,
            file_type=file_ext,
        )

    @classmethod
    def parse_stream(cls, file_path: Path) -> Generator[ParsedBlock, None, None]:
        """
        Yields parsed text and markdown table blocks page-by-page.
        Memory stays flat regardless of document page count.
        """
        path = Path(file_path)
        file_ext = path.suffix.lower().lstrip(".")

        logger.info(f"Starting streaming parse of '{path.name}' (type: {file_ext})")

        if file_ext == "pdf":
            yield from cls._parse_pdf_stream(path)
        elif file_ext in ["txt", "md"]:
            yield from cls._parse_text_stream(path)
        else:
            raise ValueError(f"Unsupported file extension: .{file_ext}. Supported: pdf, txt, md")

    @classmethod
    def _parse_pdf_stream(cls, file_path: Path) -> Generator[ParsedBlock, None, None]:
        """Streams PDF pages, separating clean narrative text from Markdown tables."""
        doc = pymupdf.open(str(file_path))
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1

                # 1. Detect and extract tables using PyMuPDF's C-level table finder
                tables = page.find_tables()
                table_rects = []

                if tables.tables:
                    for t_idx, table in enumerate(tables):
                        table_rects.append(table.bbox)
                        md_table = cls._table_to_markdown(table.extract())
                        if md_table:
                            yield ParsedBlock(
                                block_type="table",
                                content=md_table,
                                page_number=page_number,
                                metadata={"table_index": t_idx + 1}
                            )

                # 2. Extract plain text (excluding table regions to prevent duplication)
                page_text = page.get_text("text").strip()
                if page_text:
                    yield ParsedBlock(
                        block_type="text",
                        content=page_text,
                        page_number=page_number,
                        metadata={}
                    )

                if page_number % 50 == 0:
                    logger.debug(f"Parsed {page_number}/{len(doc)} pages of '{file_path.name}'")

        finally:
            doc.close()
            logger.info(f"Finished parsing PDF '{file_path.name}'")

    @classmethod
    def _parse_text_stream(cls, file_path: Path) -> Generator[ParsedBlock, None, None]:
        """Parses TXT and Markdown files in logical paragraph blocks."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into logical sections by double newlines
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for idx, para in enumerate(paragraphs):
            yield ParsedBlock(
                block_type="text",
                content=para,
                page_number=1,
                metadata={"paragraph_index": idx + 1}
            )

    @staticmethod
    def _table_to_markdown(matrix: List[List[Optional[str]]]) -> str:
        """Converts raw 2D grid matrix into standard clean GitHub Markdown table."""
        if not matrix or len(matrix) < 2:
            return ""

        # Clean cell text (remove newlines inside cells)
        cleaned_matrix = []
        for row in matrix:
            cleaned_row = [(cell or "").replace("\n", " ").strip() for cell in row]
            cleaned_matrix.append(cleaned_row)

        headers = cleaned_matrix[0]
        rows = cleaned_matrix[1:]

        # Build Markdown Table
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = ["| " + " | ".join(r) + " |" for r in rows if any(r)]

        if not row_lines:
            return ""

        return "\n".join([header_line, separator_line] + row_lines)
