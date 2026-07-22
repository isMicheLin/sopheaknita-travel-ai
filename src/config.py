"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    """Immutable settings shared by ingestion, retrieval, and generation."""

    base_dir: Path
    docs_dir: Path
    index_dir: Path
    api_key: str
    generation_model: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int

    @classmethod
    def from_environment(cls, base_dir: Path) -> "Settings":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        docs_dir = Path(os.getenv("TRAVEL_DOCS_DIR", base_dir / "data" / "travel_docs"))
        index_dir = Path(os.getenv("INDEX_DIR", base_dir / "data" / "index"))

        return cls(
            base_dir=base_dir,
            docs_dir=docs_dir,
            index_dir=index_dir,
            api_key=api_key,
            generation_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        )
