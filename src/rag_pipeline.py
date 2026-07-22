"""Coordinator for ingestion, embedding, retrieval, and generation."""

from __future__ import annotations

import hashlib
import json
import re

from .config import Settings
from .embeddings import GeminiEmbedder
from .generation import GeminiTravelGenerator
from .ingestion import DocumentChunk, chunk_documents, load_documents
from .vector_store import RetrievalResult, VectorStore


class TravelRAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = GeminiEmbedder(
            api_key=settings.api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        )
        self.generator = GeminiTravelGenerator(
            api_key=settings.api_key,
            model=settings.generation_model,
        )

        documents = load_documents(settings.docs_dir)
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.document_count = len(documents)
        self.chunk_count = len(chunks)
        fingerprint = self._fingerprint(chunks)

        cached = VectorStore.load(settings.index_dir, expected_fingerprint=fingerprint)
        if cached is not None:
            self.store = cached
        else:
            searchable_texts = [
                f"Title: {chunk.title}\nSource: {chunk.source}\n{chunk.text}"
                for chunk in chunks
            ]
            embeddings = self.embedder.embed_documents(searchable_texts)
            self.store = VectorStore(chunks=chunks, embeddings=embeddings)
            self.store.save(settings.index_dir, fingerprint=fingerprint)

    def _fingerprint(self, chunks: list[DocumentChunk]) -> str:
        payload = {
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _is_greeting(question: str) -> bool:
        normalized = re.sub(r"[^a-zA-Z ]", "", question).strip().lower()
        return normalized in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

    @staticmethod
    def _retrieval_query(question: str, history: list[dict]) -> str:
        """Add the latest user context to short follow-up questions."""
        if len(question.split()) >= 7:
            return question
        previous_user_messages = [
            item.get("content", "")
            for item in history
            if item.get("role") == "user" and item.get("content")
        ]
        if previous_user_messages:
            return f"Previous topic: {previous_user_messages[-1]}\nFollow-up: {question}"
        return question

    def ask(
        self,
        question: str,
        history: list[dict],
        top_k: int,
        min_score: float,
        answer_style: str,
    ) -> dict:
        if not question.strip():
            return {"answer": "Please enter a travel question.", "sources": []}

        if self._is_greeting(question):
            return {
                "answer": (
                    "Hello! I’m Sopheaknita Travel AI. Ask me about destinations, tour packages, "
                    "booking steps, luggage, payment, rescheduling, or refund policies."
                ),
                "sources": [],
            }

        retrieval_query = self._retrieval_query(question, history)
        query_embedding = self.embedder.embed_query(retrieval_query)
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        if not results:
            return {
                "answer": (
                    "I could not find enough relevant information in the travel agency documents to answer this safely. "
                    "Please ask about the listed destinations, packages, booking process, payment, luggage, rescheduling, or refunds."
                ),
                "sources": [],
            }

        answer = self.generator.generate(
            question=question,
            results=results,
            history=history,
            answer_style=answer_style,
        )
        return {
            "answer": answer,
            "sources": [result.to_public_dict() for result in results],
        }
