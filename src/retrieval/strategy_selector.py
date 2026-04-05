"""Retrieval strategy selector.

Decides which retrieval method to use for a given hypothesis based on its
domain, the type of available evidence, and simple heuristics.  In a
production system this could be an LLM-powered decision, but for the
pedagogical implementation we use deterministic rules to keep it transparent.

The selection logic:
- **Keyword** — when the hypothesis contains specific identifiers (error
  codes, service names, status codes) that are best matched exactly.
- **TF-IDF** — when the evidence corpus is large and unstructured (log text),
  so rare-term weighting helps surface anomalies.
- **Semantic** — when the hypothesis is a fuzzy or conceptual query that may
  not use the same terminology as the evidence.
"""

from __future__ import annotations

import re


# Patterns that suggest keyword search will be most effective
_KEYWORD_PATTERNS = re.compile(
    r"(error|exception|HTTP [0-9]{3}|status code|port \d+|"
    r"timeout|refused|denied|unreachable|evict)",
    re.IGNORECASE,
)

# Domains where TF-IDF excels (large, noisy corpora)
_TFIDF_DOMAINS = {"app-logs", "network-logs", "node-metrics"}

# Domains where evidence is highly structured
_STRUCTURED_DOMAINS = {"db-metrics", "k8s-events", "tls-certs"}


def select_strategy(
    hypothesis: dict,
    available_evidence: dict,
) -> str:
    """Select the best retrieval strategy for a hypothesis.

    Parameters
    ----------
    hypothesis:
        A hypothesis dict with at least ``text`` and ``domain`` keys.
    available_evidence:
        The evidence_sources dict ``{domain: [items]}``.

    Returns
    -------
    str
        One of ``"keyword"``, ``"tfidf"``, or ``"semantic"``.
    """
    text = hypothesis.get("text", "")
    domain = hypothesis.get("domain", "")

    # If the hypothesis text contains specific identifiers, keyword is fastest
    if _KEYWORD_PATTERNS.search(text):
        return "keyword"

    # If the evidence domains are large log corpora, TF-IDF finds rare signals
    evidence_domains = set(available_evidence.keys())
    if evidence_domains & _TFIDF_DOMAINS:
        total_items = sum(
            len(v) for k, v in available_evidence.items()
            if k in _TFIDF_DOMAINS and isinstance(v, list)
        )
        if total_items > 10:
            return "tfidf"

    # For structured data or when domain is infrastructure/database,
    # semantic search handles the conceptual gap
    if domain in ("infrastructure", "database", "security"):
        return "semantic"

    # Default: semantic search handles the widest range of queries
    return "semantic"
