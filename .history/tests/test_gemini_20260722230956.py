python
from google import genai

client = genai.Client()
interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input="Reply with: Gemini connection successful",
    store=False,
)
print(interaction.output_text)