"""Grounded response generation through the Gemini generateContent API."""

from __future__ import annotations

import random
import time

from google import genai

from .vector_store import RetrievalResult


class GeminiTravelGenerator:
    """Generate grounded answers with retry and current-model fallback support."""

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash") -> None:
        self.client = genai.Client(api_key=api_key)

        # The .env model is tried first. Current backup models follow.
        candidates = [
            model,
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
        ]
        self.models = list(dict.fromkeys(candidates))

    @staticmethod
    def _history_text(history: list[dict], limit: int = 6) -> str:
        selected = history[-limit:]
        lines: list[str] = []
        for message in selected:
            role = "User" if message.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines) or "No previous conversation."

    @staticmethod
    def _is_temporary_error(exc: Exception) -> bool:
        """Return True for temporary capacity, quota, or service errors."""
        message = str(exc).lower()
        markers = (
            "429",
            "503",
            "resource_exhausted",
            "unavailable",
            "high demand",
            "rate limit",
            "temporarily overloaded",
            "temporarily busy",
            "running out of capacity",
            "service unavailable",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _is_model_unavailable(exc: Exception) -> bool:
        """Return True when a model is retired, inaccessible, or unsupported."""
        message = str(exc).lower()
        markers = (
            "404",
            "not_found",
            "not found",
            "no longer available",
            "not supported for generatecontent",
            "model is not available",
        )
        return any(marker in message for marker in markers)

    def _generate_with_model(self, model: str, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            raise RuntimeError(f"Gemini model {model} returned an empty response.")
        return answer

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        history: list[dict],
        answer_style: str,
    ) -> str:
        source_blocks: list[str] = []
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

        errors: list[str] = []

        for model in self.models:
            for attempt in range(2):
                try:
                    return self._generate_with_model(model=model, prompt=prompt)
                except Exception as exc:
                    errors.append(f"{model}: {exc}")

                    # A retired or inaccessible model should be skipped immediately.
                    if self._is_model_unavailable(exc):
                        break

                    # Retry temporary errors once, then move to the next model.
                    if self._is_temporary_error(exc):
                        if attempt == 0:
                            time.sleep(1.5 + random.uniform(0.0, 0.7))
                            continue
                        break

                    # Authentication, malformed request, or other permanent errors.
                    raise

        final_detail = errors[-1] if errors else "No Gemini model returned a response."
        raise RuntimeError(
            "No configured Gemini generation model is available right now. "
            "The app tried current primary and backup models. "
            f"Last error: {final_detail}"
        )