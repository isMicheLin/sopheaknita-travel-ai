"""Optional live smoke test for the Gemini API integration."""

import os

import pytest
from google import genai


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is not set; skipping live Gemini smoke test.",
)
def test_gemini_connection_smoke() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input="Reply with: Gemini connection successful",
        store=False,
    )

    assert interaction.output_text
    assert "Gemini connection successful" in interaction.output_text
