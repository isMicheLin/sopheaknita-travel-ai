"""Command-line helper to build or refresh the document embedding cache."""

from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.config import Settings  # noqa: E402
from src.rag_pipeline import TravelRAGPipeline  # noqa: E402


if __name__ == "__main__":
    settings = Settings.from_environment(ROOT)
    pipeline = TravelRAGPipeline(settings)
    print(
        f"Index ready: {pipeline.document_count} documents, "
        f"{pipeline.chunk_count} chunks."
    )
