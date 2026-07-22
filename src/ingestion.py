"""Document loading and paragraph-aware chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import re
from typing import Iterable

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip().lstrip("#").strip()
        if candidate:
            return candidate[:120]
    return path.stem.replace("_", " ").title()


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"Page {page_number}\n{page_text}")
    return "\n\n".join(pages)


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            values = [f"{key}: {value}" for key, value in row.items() if value]
            if values:
                rows.append(f"Row {row_number}\n" + "\n".join(values))
    return "\n\n".join(rows)


def read_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.name}")

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".csv":
        text = _read_csv(path)
    else:
        text = path.read_text(encoding="utf-8-sig", errors="replace")

    text = _clean_text(text)
    if not text:
        raise ValueError(f"No readable text found in {path.name}")

    return Document(source=path.name, title=_title_from_text(path, text), text=text)


def load_documents(folder: Path) -> list[Document]:
    if not folder.exists():
        raise FileNotFoundError(f"Document folder does not exist: {folder}")

    documents: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(read_document(path))

    if not documents:
        raise ValueError(f"No supported documents were found in {folder}")
    return documents


def _split_long_segment(segment: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(segment):
        end = min(start + chunk_size, len(segment))
        if end < len(segment):
            boundary = max(
                segment.rfind(". ", start, end),
                segment.rfind("? ", start, end),
                segment.rfind("! ", start, end),
                segment.rfind("\n", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = segment[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(segment):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(document: Document, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    """Split text by paragraphs, then apply an overlapping window when needed."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
    merged: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            merged.append(current)
            current = ""

        if len(paragraph) > chunk_size:
            merged.extend(_split_long_segment(paragraph, chunk_size, overlap))
        else:
            current = paragraph

    if current:
        merged.append(current)

    chunks: list[DocumentChunk] = []
    for position, text in enumerate(merged):
        digest = hashlib.sha1(
            f"{document.source}:{position}:{text}".encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(
            DocumentChunk(
                chunk_id=digest,
                source=document.source,
                title=document.title,
                text=text,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, overlap: int
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap))
    if not chunks:
        raise ValueError("Chunking produced no searchable text.")
    return chunks
