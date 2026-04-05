"""Keyword retrieval strategy — exact substring and regex matching.

Best suited for known error patterns: specific error codes, service names,
HTTP status codes, or exact log fragments.  Fast and precise, but cannot
handle synonyms or semantic similarity.

In production this would map to a ``grep`` against a log store or an
Elasticsearch ``match_phrase`` query.
"""

from __future__ import annotations

import re


def keyword_search(
    query: str,
    evidence_corpus: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Search evidence by keyword / substring matching.

    Splits the query into individual keywords and scores each evidence
    item by how many keywords it contains (case-insensitive).  Returns
    up to *top_k* results sorted by match count descending.
    """
    keywords = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]

    scored: list[tuple[int, dict]] = []
    for item in evidence_corpus:
        text = item.get("text", "").lower()
        matches = sum(1 for kw in keywords if kw in text)
        if matches > 0:
            scored.append((matches, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]
