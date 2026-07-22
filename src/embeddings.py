"""Gemini embedding client with batching and retry handling."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from google import genai
from google.genai import types


class GeminiEmbedder:
    """Generate real semantic embeddings for documents and queries."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        batch_size: int = 32,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def _request(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=list(texts),
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimensions,
                    ),
                )
                vectors = [embedding.values for embedding in result.embeddings]
                return np.asarray(vectors, dtype=np.float32)
            except Exception as exc:  # API/network errors vary by SDK version
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Gemini embedding request failed: {last_error}")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batches.append(self._request(batch, task_type="RETRIEVAL_DOCUMENT"))
        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Query text cannot be empty.")
        return self._request([text], task_type="RETRIEVAL_QUERY")[0]
