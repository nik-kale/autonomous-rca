"""Tests for the Analysis Agent — investigation synthesis.

All tests mock the OpenAI client so they run without an API key.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.analysis import synthesize_investigation
from src.state import CAUSES, ROOT_CAUSE


_MOCK_SYNTHESIS = json.dumps({
    "root_cause": "Disk pressure on worker-3 caused pod eviction",
    "reasoning_path": [
        "Service became unreachable",
        "Pod was evicted",
        "DiskPressure=True on worker-3",
    ],
    "resolution_steps": [
        "Clear disk space",
        "Add monitoring",
    ],
    "confidence": 0.92,
    "summary": "Disk pressure caused the outage.",
})


def _mock_openai_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


@patch("src.llm.OpenAI")
def test_produces_root_cause(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_SYNTHESIS)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Checkout service unreachable",
        "hypotheses": [
            {"id": "h1", "text": "Disk pressure", "status": "root_cause", "domain": "infrastructure", "iteration": 0},
            {"id": "h2", "text": "Network issue", "status": "rejected", "domain": "network", "iteration": 0},
        ],
        "evaluations": [
            {"hypothesis_id": "h1", "confirmed": True, "confidence": 0.92},
        ],
        "evidence": {},
        "nodes": [
            {"id": "problem_root", "type": "problem", "text": "Checkout service unreachable", "iteration": 0},
        ],
        "edges": [],
        "iteration": 2,
        "max_iterations": 5,
        "converged": True,
        "root_cause": None,
    }

    result = synthesize_investigation(state)

    assert result["root_cause"] is not None
    assert "Disk pressure" in result["root_cause"]
    assert result["converged"] is True


@patch("src.llm.OpenAI")
def test_emits_synthesis_node(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_SYNTHESIS)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Checkout service unreachable",
        "hypotheses": [
            {"id": "h1", "text": "Disk pressure", "status": "root_cause", "domain": "infrastructure", "iteration": 0},
        ],
        "evaluations": [],
        "evidence": {},
        "nodes": [],
        "edges": [],
        "iteration": 2,
        "max_iterations": 5,
        "converged": True,
        "root_cause": None,
    }

    result = synthesize_investigation(state)

    synthesis_nodes = [n for n in result["nodes"] if n["id"] == "synthesis"]
    assert len(synthesis_nodes) == 1
    assert synthesis_nodes[0]["type"] == ROOT_CAUSE
    assert synthesis_nodes[0]["confidence"] == 0.92


@patch("src.llm.OpenAI")
def test_links_to_confirmed_hypothesis(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_SYNTHESIS)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Checkout service unreachable",
        "hypotheses": [
            {"id": "h1", "text": "Disk pressure", "status": "root_cause", "domain": "infrastructure", "iteration": 0},
            {"id": "h2", "text": "Network issue", "status": "rejected", "domain": "network", "iteration": 0},
        ],
        "evaluations": [],
        "evidence": {},
        "nodes": [],
        "edges": [],
        "iteration": 2,
        "max_iterations": 5,
        "converged": True,
        "root_cause": None,
    }

    result = synthesize_investigation(state)

    causes_edges = [e for e in result["edges"] if e["type"] == CAUSES]
    assert len(causes_edges) == 1
    assert causes_edges[0]["from_id"] == "h1"
    assert causes_edges[0]["to_id"] == "synthesis"
