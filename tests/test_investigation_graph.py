"""Tests for the full investigation pipeline — end-to-end graph validity.

Mocks all LLM calls to produce a deterministic investigation that exercises
the full loop: diagnose → retrieve → evaluate → (loop) → analyze.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.graph.builder import build_investigation_graph, should_continue
from src.state import (
    CAUSES,
    CONTRADICTS,
    DEPENDS_ON,
    EVIDENCE,
    GENERATED_FROM,
    HYPOTHESIS,
    PROBLEM,
    REJECTED,
    ROOT_CAUSE,
    SUPPORTS,
)


# --- Mock LLM responses for a deterministic 2-iteration investigation ---
# Hypothesis IDs are prefixed by generate_hypotheses: iteration 0 → i0_*, iteration 1 → i1_*

_ITER1_HYPOTHESES = json.dumps([
    {"id": "h1", "text": "Network connectivity failure", "domain": "network"},
    {"id": "h2", "text": "Pod evicted due to resource pressure", "domain": "infrastructure"},
    {"id": "h3", "text": "Application crash in checkout container", "domain": "application"},
])

_ITER1_EVALUATION = json.dumps([
    {"hypothesis_id": "i0_h1", "confirmed": False, "cause_of": None, "confidence": 0.1,
     "reasoning": "Network logs show everything healthy"},
    {"hypothesis_id": "i0_h2", "confirmed": True, "cause_of": "i0_h3", "confidence": 0.7,
     "reasoning": "Pod eviction explains checkout failure but needs deeper investigation"},
    {"hypothesis_id": "i0_h3", "confirmed": False, "cause_of": None, "confidence": 0.15,
     "reasoning": "No crash loops found in container logs"},
])

_ITER2_HYPOTHESES = json.dumps([
    {"id": "h4", "text": "Disk pressure on worker node caused eviction", "domain": "infrastructure"},
    {"id": "h5", "text": "Memory limits exceeded on worker node", "domain": "infrastructure"},
])

_ITER2_EVALUATION = json.dumps([
    {"hypothesis_id": "i1_h4", "confirmed": True, "cause_of": None, "confidence": 0.92,
     "reasoning": "DiskPressure=True on worker-3 directly caused pod eviction"},
    {"hypothesis_id": "i1_h5", "confirmed": False, "cause_of": None, "confidence": 0.1,
     "reasoning": "Memory usage was within normal range"},
])

_SYNTHESIS = json.dumps({
    "root_cause": "Disk pressure on worker-3 from ephemeral storage exhaustion",
    "reasoning_path": [
        "Checkout service became unreachable",
        "Pod checkout-pod-7b was evicted from worker-3",
        "Worker-3 had DiskPressure=True condition",
        "Disk usage reached 94%, exceeding 90% threshold",
    ],
    "resolution_steps": [
        "Clear ephemeral storage on worker-3",
        "Increase ephemeral storage limits",
        "Add monitoring alert for disk usage above 80%",
    ],
    "confidence": 0.92,
    "summary": "The checkout service outage was caused by disk pressure on node worker-3.",
})


def _build_mock_client(responses: list[str]) -> MagicMock:
    """Build a mock OpenAI client that returns responses in order."""
    mock_client = MagicMock()
    mock_responses = []
    for content in responses:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        mock_responses.append(mock_resp)
    mock_client.chat.completions.create.side_effect = mock_responses
    return mock_client


class TestShouldContinue:
    def test_returns_analyze_when_converged(self):
        state = {"converged": True, "iteration": 1, "max_iterations": 5, "hypotheses": []}
        assert should_continue(state) == "analyze"

    def test_returns_analyze_at_max_iterations(self):
        state = {"converged": False, "iteration": 5, "max_iterations": 5, "hypotheses": []}
        assert should_continue(state) == "analyze"

    def test_returns_analyze_when_all_resolved(self):
        state = {
            "converged": False,
            "iteration": 1,
            "max_iterations": 5,
            "hypotheses": [
                {"id": "h1", "status": "rejected"},
                {"id": "h2", "status": "root_cause"},
            ],
        }
        assert should_continue(state) == "analyze"

    def test_returns_diagnose_when_refinable(self):
        state = {
            "converged": False,
            "iteration": 1,
            "max_iterations": 5,
            "hypotheses": [
                {"id": "h1", "status": "rejected"},
                {"id": "h2", "status": "active"},
            ],
        }
        assert should_continue(state) == "diagnose"


class TestFullPipeline:
    @patch("src.llm.OpenAI")
    def test_produces_valid_investigation_graph(self, mock_openai_cls):
        mock_openai_cls.return_value = _build_mock_client([
            _ITER1_HYPOTHESES,
            _ITER1_EVALUATION,
            _ITER2_HYPOTHESES,
            _ITER2_EVALUATION,
            _SYNTHESIS,
        ])

        from src.evidence.mock_data import KUBERNETES_DISK_PRESSURE

        app = build_investigation_graph()
        result = app.invoke({
            "problem_statement": KUBERNETES_DISK_PRESSURE["problem_statement"],
            "evidence_sources": KUBERNETES_DISK_PRESSURE["evidence_sources"],
            "evidence": {},
            "hypotheses": [],
            "evaluations": [],
            "nodes": [],
            "edges": [],
            "iteration": 0,
            "max_iterations": 5,
            "converged": False,
            "root_cause": None,
        })

        assert len(result["nodes"]) > 0
        assert len(result["edges"]) > 0

        node_types = {n["type"] for n in result["nodes"]}
        assert PROBLEM in node_types
        assert HYPOTHESIS in node_types

        edge_types = {e["type"] for e in result["edges"]}
        assert GENERATED_FROM in edge_types

        assert result["converged"] is True
        assert result["root_cause"] is not None

    @patch("src.llm.OpenAI")
    def test_graph_has_valid_node_references(self, mock_openai_cls):
        mock_openai_cls.return_value = _build_mock_client([
            _ITER1_HYPOTHESES,
            _ITER1_EVALUATION,
            _ITER2_HYPOTHESES,
            _ITER2_EVALUATION,
            _SYNTHESIS,
        ])

        from src.evidence.mock_data import KUBERNETES_DISK_PRESSURE

        app = build_investigation_graph()
        result = app.invoke({
            "problem_statement": KUBERNETES_DISK_PRESSURE["problem_statement"],
            "evidence_sources": KUBERNETES_DISK_PRESSURE["evidence_sources"],
            "evidence": {},
            "hypotheses": [],
            "evaluations": [],
            "nodes": [],
            "edges": [],
            "iteration": 0,
            "max_iterations": 5,
            "converged": False,
            "root_cause": None,
        })

        node_ids = {n["id"] for n in result["nodes"]}

        for edge in result["edges"]:
            from_id = edge["from_id"]
            to_id = edge["to_id"]
            assert from_id in node_ids or from_id in {h["id"] for h in result["hypotheses"]}, (
                f"Edge from_id '{from_id}' not found in nodes or hypotheses"
            )

    @patch("src.llm.OpenAI")
    def test_no_self_loop_edges_in_pipeline(self, mock_openai_cls):
        """End-to-end check that no edge has from_id == to_id."""
        mock_openai_cls.return_value = _build_mock_client([
            _ITER1_HYPOTHESES,
            _ITER1_EVALUATION,
            _ITER2_HYPOTHESES,
            _ITER2_EVALUATION,
            _SYNTHESIS,
        ])

        from src.evidence.mock_data import KUBERNETES_DISK_PRESSURE

        app = build_investigation_graph()
        result = app.invoke({
            "problem_statement": KUBERNETES_DISK_PRESSURE["problem_statement"],
            "evidence_sources": KUBERNETES_DISK_PRESSURE["evidence_sources"],
            "evidence": {},
            "hypotheses": [],
            "evaluations": [],
            "nodes": [],
            "edges": [],
            "iteration": 0,
            "max_iterations": 5,
            "converged": False,
            "root_cause": None,
        })

        for edge in result["edges"]:
            assert edge["from_id"] != edge["to_id"], (
                f"Self-loop: {edge['from_id']} → {edge['to_id']} (type={edge['type']})"
            )

    @patch("src.llm.OpenAI")
    def test_evidence_corpus_preserved_across_iterations(self, mock_openai_cls):
        """The original evidence_sources must remain intact through all iterations."""
        mock_openai_cls.return_value = _build_mock_client([
            _ITER1_HYPOTHESES,
            _ITER1_EVALUATION,
            _ITER2_HYPOTHESES,
            _ITER2_EVALUATION,
            _SYNTHESIS,
        ])

        from src.evidence.mock_data import KUBERNETES_DISK_PRESSURE

        original_sources = KUBERNETES_DISK_PRESSURE["evidence_sources"]
        app = build_investigation_graph()
        result = app.invoke({
            "problem_statement": KUBERNETES_DISK_PRESSURE["problem_statement"],
            "evidence_sources": original_sources,
            "evidence": {},
            "hypotheses": [],
            "evaluations": [],
            "nodes": [],
            "edges": [],
            "iteration": 0,
            "max_iterations": 5,
            "converged": False,
            "root_cause": None,
        })

        assert result["evidence_sources"] == original_sources
