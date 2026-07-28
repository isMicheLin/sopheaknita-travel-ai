"""Grounded response generation through the Gemini Interactions API."""

from __future__ import annotations

from google import genai

from .vector_store import RetrievalResult


class GeminiTravelGenerator:
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @staticmethod
    def _history_text(history: list[dict], limit: int = 6) -> str:
        selected = history[-limit:]
        lines: list[str] = []
        for message in selected:
            role = "User" if message.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines) or "No previous conversation."

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        history: list[dict],
        answer_style: str,
    ) -> str:
        source_blocks = []
        for number, result in enumerate(results, start=1):
            source_blocks.append(
                f"[{number}] Title: {result.chunk.title}\n"
                f"File: {result.chunk.source}\n"
                f"Text: {result.chunk.text}"
            )

        style_instruction = (
            "Keep the answer under 140 words unless the user asks for more detail."
            if answer_style == "Concise"
            else "Give a structured answer with enough detail for travel planning."
        )

        prompt = f"""
You are Sopheaknita Travel AI, a friendly and professional travel agency assistant.

Rules:
1. Answer the current question using only the retrieved source text below.
2. Cite factual statements with bracket citations such as [1] or [2].
3. Never invent prices, schedules, policies, availability, visa rules, or package details.
4. When the sources do not contain enough information, clearly say what information is missing.
5. Treat the documents as agency information, not universal travel law.
6. Respond in the same language as the user's question when practical.
7. {style_instruction}

Recent conversation:
{self._history_text(history)}

Retrieved sources:
{chr(10).join(source_blocks)}

Current user question:
{question}

Write the final answer now. Do not include a separate references section because the application displays the retrieved chunks below the answer.
""".strip()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
        return answer
