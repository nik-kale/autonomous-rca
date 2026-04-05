"""Semantic retrieval strategy — embedding-based similarity search.

Best suited for fuzzy, intent-based queries where the user's phrasing may
not match the exact terminology in the evidence.  Handles synonyms and
conceptual similarity (e.g., "disk full" matches "storage exhaustion").

Uses OpenAI embeddings when an API key is available, falling back to a
local sentence-transformers model for offline / CI usage.

In production this would map to a vector database query (Pinecone, Weaviate,
pgvector, etc.).
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _embed_openai(texts: list[str]) -> np.ndarray:
    """Embed texts using the OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([item.embedding for item in response.data])


def _embed_local(texts: list[str]) -> np.ndarray:
    """Embed texts using a local sentence-transformers model.

    Falls back to TF-IDF vectors if the model cannot be loaded (e.g. no
    network, missing model cache, or sentence-transformers not installed).
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, convert_to_numpy=True)
    except Exception:
        logger.warning(
            "sentence-transformers unavailable; falling back to TF-IDF embeddings",
            exc_info=True,
        )
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between a single vector and a matrix."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    if a_norm.ndim == 1:
        a_norm = a_norm.reshape(1, -1)
    return (a_norm @ b_norm.T).flatten()


def semantic_search(
    query: str,
    evidence_corpus: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Retrieve evidence by semantic similarity to the query.

    Embeds the query and all evidence texts, then returns the *top_k* most
    similar items by cosine similarity.
    """
    if not evidence_corpus:
        return []

    texts = [item.get("text", "") for item in evidence_corpus]
    all_texts = [query] + texts

    if os.environ.get("OPENAI_API_KEY"):
        try:
            embeddings = _embed_openai(all_texts)
        except Exception:
            logger.warning(
                "OpenAI embedding call failed; falling back to local model",
                exc_info=True,
            )
            embeddings = _embed_local(all_texts)
    else:
        embeddings = _embed_local(all_texts)

    query_embedding = embeddings[0]
    corpus_embeddings = embeddings[1:]

    similarities = _cosine_similarity(query_embedding, corpus_embeddings)
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [evidence_corpus[i] for i in ranked_indices if similarities[i] > 0]
