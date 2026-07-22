# Evaluation Write-Up Template

## Method

Test the system with 8 queries from `evaluation_queries.csv`. For each query, record:

- Whether the correct source appeared in the top-k results
- Whether the answer used only retrieved information
- Whether citations matched the source text
- Whether the answer was clear and useful
- Whether the system refused unsupported questions
- Approximate response time

## Results table

| Query ID | Retrieval result | Answer quality | Citation quality | Latency | Notes |
|---|---|---|---|---|---|
| Q1 |  |  |  |  |  |
| Q2 |  |  |  |  |  |
| Q3 |  |  |  |  |  |
| Q4 |  |  |  |  |  |
| Q5 |  |  |  |  |  |
| Q6 |  |  |  |  |  |
| Q7 |  |  |  |  |  |
| Q8 |  |  |  |  |  |

## Design decisions to justify

1. Gemini embeddings replace TF-IDF because semantic vectors match meaning and synonyms, not only shared words.
2. Paragraph-aware chunks preserve topic boundaries, while 150-character overlap reduces context loss between chunks.
3. Cosine similarity directly extends the Week 14 ranking method and is suitable for a small academic dataset.
4. A 0.30 relevance threshold supports graceful failure and reduces unsupported answers.
5. The generator receives retrieved text and explicit citation rules to keep answers grounded.

## Limitations and next steps

Write 1 to 2 paragraphs about failed queries, weak retrieval cases, API latency, document quality, and planned improvements.
