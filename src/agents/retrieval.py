"""Retrieval Agent — evidence gathering.

For each active hypothesis the agent selects an appropriate retrieval strategy
(keyword, TF-IDF, or semantic search) and queries the evidence corpus.  In
this pedagogical implementation the corpus comes from mock data; in a
production system each strategy would call a different backend (log store,
metrics API, knowledge base, etc.).

The agent emits ``evidence`` nodes and ``supports``/``contradicts`` edges
into the Investigation Graph.  At this stage the relationship direction is
tentative — the Evaluation Agent will confirm or revise it.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.retrieval.strategy_selector import select_strategy
from src.retrieval.keyword import keyword_search
from src.retrieval.tfidf import tfidf_search
from src.retrieval.semantic import semantic_search
from src.state import EVIDENCE as EVIDENCE_TYPE, SUPPORTS, InvestigationState

_STRATEGY_DISPATCH = {
    "keyword": keyword_search,
    "tfidf": tfidf_search,
    "semantic": semantic_search,
}


def _flatten_evidence_sources(evidence_sources: dict) -> list[dict]:
    """Flatten ``{domain: [items]}`` into a single corpus list."""
    corpus: list[dict] = []
    for domain, items in evidence_sources.items():
        if isinstance(items, list):
            for item in items:
                corpus.append({**item, "domain": domain})
    return corpus


def retrieve_evidence(state: InvestigationState) -> dict[str, Any]:
    """Retrieve relevant evidence for each active hypothesis.

    Returns a partial state update containing:
    - ``evidence`` — dict mapping hypothesis ids to retrieved items
    - ``nodes`` — new evidence nodes for the Investigation Graph
    - ``edges`` — tentative ``supports`` edges (to be confirmed by evaluator)
    """
    hypotheses = state.get("hypotheses", [])
    existing_evidence: dict = dict(state.get("evidence", {}))
    evidence_sources: dict = state.get("evidence_sources", {})

    corpus = _flatten_evidence_sources(evidence_sources)

    new_nodes: list[dict] = []
    new_edges: list[dict] = []
    updated_evidence: dict[str, list[dict]] = {}

    for h in hypotheses:
        if h.get("status") in ("rejected", "root_cause"):
            continue

        h_id = h["id"]

        if h_id in existing_evidence and isinstance(existing_evidence[h_id], list):
            updated_evidence[h_id] = existing_evidence[h_id]
            continue

        strategy_name = select_strategy(h, evidence_sources)
        search_fn = _STRATEGY_DISPATCH.get(strategy_name, keyword_search)

        # In production: this is where you'd call a log aggregation
        # platform, metrics API, or vector database instead of the mock
        # corpus.
        results = search_fn(h["text"], corpus, top_k=5)

        updated_evidence[h_id] = results

        for item in results:
            ev_id = f"ev_{uuid.uuid4().hex[:8]}"
            new_nodes.append(
                {
                    "id": ev_id,
                    "type": EVIDENCE_TYPE,
                    "text": item.get("text", "")[:200],
                    "iteration": state.get("iteration", 0),
                    "confidence": None,
                }
            )
            new_edges.append(
                {
                    "from_id": ev_id,
                    "to_id": h_id,
                    "type": SUPPORTS,
                    "iteration": state.get("iteration", 0),
                }
            )

    return {
        "evidence": updated_evidence,
        "nodes": new_nodes,
        "edges": new_edges,
    }
