"""In-memory cosine-similarity vector index with disk persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .ingestion import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float

    def to_public_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "score": self.score,
        }


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class VectorStore:
    def __init__(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")
        if len(chunks) == 0:
            raise ValueError("The vector store cannot be empty.")
        self.chunks = chunks
        self.embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        min_score: float = 0.30,
    ) -> list[RetrievalResult]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = _normalize(query)[0]
        scores = self.embeddings @ query
        ranked_indices = np.argsort(scores)[::-1]

        results: list[RetrievalResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue
            results.append(RetrievalResult(chunk=self.chunks[index], score=score))
            if len(results) >= top_k:
                break
        return results

    def save(self, index_dir: Path, fingerprint: str) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_dir / "embeddings.npz", embeddings=self.embeddings)
        metadata = {
            "fingerprint": fingerprint,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        (index_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path, expected_fingerprint: str) -> "VectorStore | None":
        metadata_path = index_dir / "metadata.json"
        embeddings_path = index_dir / "embeddings.npz"
        if not metadata_path.exists() or not embeddings_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != expected_fingerprint:
            return None

        chunks = [DocumentChunk(**item) for item in metadata["chunks"]]
        embeddings = np.load(embeddings_path)["embeddings"]
        return cls(chunks=chunks, embeddings=embeddings)
