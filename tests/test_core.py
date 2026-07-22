"""Offline tests for chunking and cosine retrieval. No API key is required."""

import numpy as np

from src.ingestion import Document, DocumentChunk, chunk_document
from src.vector_store import VectorStore


def test_chunking_produces_metadata() -> None:
    document = Document(
        source="test.txt",
        title="Test Travel Guide",
        text="Paragraph one about buses.\n\nParagraph two about hotels and food.",
    )
    chunks = chunk_document(document, chunk_size=45, overlap=8)
    assert chunks
    assert all(chunk.source == "test.txt" for chunk in chunks)
    assert all(chunk.title == "Test Travel Guide" for chunk in chunks)


def test_vector_store_ranks_closest_vector() -> None:
    chunks = [
        DocumentChunk(chunk_id="a", source="a.txt", title="A", text="Bus travel"),
        DocumentChunk(chunk_id="b", source="b.txt", title="B", text="Beach travel"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(chunks, embeddings)
    results = store.search(np.asarray([0.9, 0.1]), top_k=1, min_score=0.0)
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
