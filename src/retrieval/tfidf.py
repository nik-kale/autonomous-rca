"""TF-IDF retrieval strategy — rare pattern detection in log corpora.

Best suited for large volumes of unstructured log text where the signal is
a *rare* pattern buried in noise.  TF-IDF naturally up-weights unusual terms
and down-weights common boilerplate, making it effective for anomaly-style
retrieval.

In production this would map to an Elasticsearch ``more_like_this`` query or
a custom TF-IDF index over recent log windows.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_search(
    query: str,
    evidence_corpus: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Rank evidence by TF-IDF cosine similarity to the query.

    Fits a TF-IDF vectorizer on the corpus, transforms both the query and
    the corpus, then returns the *top_k* most similar items.
    """
    if not evidence_corpus:
        return []

    texts = [item.get("text", "") for item in evidence_corpus]
    texts.append(query)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    query_vec = tfidf_matrix[-1]
    corpus_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vec, corpus_vecs).flatten()
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [evidence_corpus[i] for i in ranked_indices if similarities[i] > 0]
