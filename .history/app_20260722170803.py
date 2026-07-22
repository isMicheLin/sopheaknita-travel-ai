y"""Streamlit interface for the Sopheaknita Travel AI RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import Settings
from src.rag_pipeline import TravelRAGPipeline

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Sopheaknita Travel AI",
    page_icon="✈️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1100px; padding-top: 2rem;}
      .source-score {font-size: 0.85rem; opacity: 0.75;}
      [data-testid="stSidebar"] {min-width: 310px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the travel knowledge base...")
def get_pipeline(settings: Settings) -> TravelRAGPipeline:
    """Create and cache the RAG pipeline for the current configuration."""
    return TravelRAGPipeline(settings=settings)


def reset_chat() -> None:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    """Render retrieved chunks with source names and similarity scores."""
    if not sources:
        return

    with st.expander(f"Retrieved sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{index}] {source['title']}**  \n"
                f"<span class='source-score'>File: {source['source']} · "
                f"Similarity: {source['score']:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.write(source["text"])
            if index < len(sources):
                st.divider()


st.title("✈️ Sopheaknita Travel AI")
st.caption(
    "A ChatGPT-style travel agency assistant powered by Gemini and retrieval-augmented generation."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Number of source chunks",
        min_value=2,
        max_value=8,
        value=4,
        help="Controls how many relevant document chunks are sent to Gemini.",
    )
    min_score = st.slider(
        "Minimum similarity score",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Higher values make the bot stricter when deciding whether the documents answer a question.",
    )
    answer_style = st.selectbox(
        "Answer style",
        options=["Concise", "Detailed"],
        index=0,
    )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    api_ready = bool(os.getenv("GEMINI_API_KEY"))
    if api_ready:
        st.success("Gemini API key connected")
    else:
        st.error("GEMINI_API_KEY is missing")
        st.caption("Copy .env.example to .env, then paste your Google AI Studio API key.")

if "messages" not in st.session_state:
    reset_chat()

if not os.getenv("GEMINI_API_KEY"):
    st.info(
        "Set up the API key first. Open GOOGLE_AI_STUDIO_GUIDE.md in the project folder and follow the steps."
    )
    st.stop()

try:
    settings = Settings.from_environment(BASE_DIR)
    pipeline = get_pipeline(settings)
except Exception as exc:
    st.error(f"The knowledge base could not be prepared: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("Knowledge base")
    st.metric("Documents", pipeline.document_count)
    st.metric("Chunks", pipeline.chunk_count)
    st.caption(f"Embedding model: {settings.embedding_model}")
    st.caption(f"Generation model: {settings.generation_model}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("Ask about destinations, packages, booking, refunds, luggage, or travel planning...")

if question:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question.strip(), "sources": []}
    )

    with st.chat_message("user"):
        st.markdown(question.strip())

    with st.chat_message("assistant"):
        with st.spinner("Searching the travel documents and preparing your answer..."):
            try:
                response = pipeline.ask(
                    question=question.strip(),
                    history=st.session_state.messages[:-1],
                    top_k=top_k,
                    min_score=min_score,
                    answer_style=answer_style,
                )
            except Exception as exc:
                response = {
                    "answer": (
                        "I could not complete the request because the Gemini API returned an error. "
                        f"Please check your API key, quota, and internet connection. Details: {exc}"
                    ),
                    "sources": [],
                }

        st.markdown(response["answer"])
        render_sources(response["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        }
    )
