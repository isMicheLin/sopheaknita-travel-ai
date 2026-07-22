# Connect the project to Google AI Studio

## 1. Create the Gemini API key

1. Sign in to Google AI Studio.
2. Open the API Keys page.
3. Select Create API key.
4. Choose or create a Google Cloud project.
5. Copy the generated key.
6. Apply the recommended key restrictions in AI Studio or Google Cloud.

API key page:
https://aistudio.google.com/app/apikey

## 2. Add the key to this project

1. Copy `.env.example` and rename the copy to `.env`.
2. Open `.env`.
3. Replace the placeholder:

```env
GEMINI_API_KEY=your_real_key_here
```

Do not put quotation marks around the key. Do not commit `.env` to GitHub. The `.gitignore` file already excludes it.

## 3. Install the official SDK

```bash
pip install -U google-genai
```

This project uses the current Google GenAI Python SDK and the Gemini Interactions API.

## 4. Test the connection

Create a temporary file named `test_gemini.py`:

```python
from google import genai

client = genai.Client()
interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input="Reply with: Gemini connection successful",
    store=False,
)
print(interaction.output_text)
```

Run it:

```bash
python test_gemini.py
```

Expected result:

```text
Gemini connection successful
```

Delete the temporary test file after the connection works.

## 5. Build the travel knowledge index

```bash
python scripts/build_index.py
```

The script reads the travel documents, splits them into chunks, creates semantic embeddings, and saves a local cache in `data/index/`.

## 6. Start the chatbot

```bash
streamlit run app.py
```

## Common problems

### GEMINI_API_KEY is missing

Confirm the file is named exactly `.env`, not `.env.txt`. Restart the terminal and Streamlit after editing it.

### 401 or invalid API key

Create a new key in Google AI Studio. Remove spaces before or after the key.

### 429 or quota error

Wait for the quota window to reset, lower the document count during testing, or review the project quota and billing settings.

### Model not found

Check the current model list in Google AI Studio. Update `GEMINI_MODEL` in `.env` to a model available to your project.

### Documents changed but old answers appear

Stop Streamlit and run:

```bash
python scripts/build_index.py
```

The index fingerprint normally detects changes automatically. You may also delete files inside `data/index/` and restart the app.
