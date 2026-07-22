# Project Notes

This starter was designed from two supplied files:

1. The CS382 final brief requires a RAG search system with ingestion, chunking, real embeddings, top-k retrieval, LLM generation, visible citations, graceful failure, a working interface, evaluation, and documentation.
2. The Week 14 Python sample uses TF-IDF and cosine similarity for a travel recommender. This project preserves the cosine-similarity ranking idea and upgrades TF-IDF to Gemini semantic embeddings.

## Presentation explanation

The system has four separable layers:

- Ingestion: reads files and creates paragraph-aware chunks
- Retrieval: embeds chunks and ranks them with cosine similarity
- Generation: asks Gemini to answer only from retrieved chunks
- Interface: shows chat answers, source text, file names, and scores

## Important submission action

Replace the fictional demo files with real, verifiable travel documents before the final presentation. Keep at least 20 documents or enough content to create an equivalent chunked volume.
