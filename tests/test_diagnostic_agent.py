"""Tests for the Diagnostic Agent — hypothesis generation.

All tests mock the OpenAI client so they run without an API key.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.diagnostic import generate_hypotheses
from src.state import HYPOTHESIS, PROBLEM


_MOCK_HYPOTHESES = json.dumps([
    {
        "id": "h1",
        "text": "Network connectivity issue between gateway and backend pods",
        "domain": "network",
    },
    {
        "id": "h2",
        "text": "Kubernetes pod eviction due to resource limits",
        "domain": "infrastructure",
    },
    {
        "id": "h3",
        "text": "Application crash loop in checkout service container",
        "domain": "application",
    },
])


def _mock_openai_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


@patch("src.agents.diagnostic.OpenAI")
def test_generates_hypotheses_on_first_iteration(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_HYPOTHESES)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Checkout service unreachable",
        "hypotheses": [],
        "evaluations": [],
        "evidence": {},
        "nodes": [],
        "edges": [],
        "iteration": 0,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = generate_hypotheses(state)

    assert len(result["hypotheses"]) == 3
    assert all(h["status"] == "active" for h in result["hypotheses"])
    assert result["iteration"] == 1


@patch("src.agents.diagnostic.OpenAI")
def test_emits_problem_node_on_first_iteration(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_HYPOTHESES)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [],
        "evaluations": [],
        "evidence": {},
        "nodes": [],
        "edges": [],
        "iteration": 0,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = generate_hypotheses(state)

    node_types = [n["type"] for n in result["nodes"]]
    assert PROBLEM in node_types
    assert node_types.count(HYPOTHESIS) == 3


@patch("src.agents.diagnostic.OpenAI")
def test_emits_generated_from_edges(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_HYPOTHESES)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [],
        "evaluations": [],
        "evidence": {},
        "nodes": [],
        "edges": [],
        "iteration": 0,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = generate_hypotheses(state)

    assert len(result["edges"]) == 3
    for edge in result["edges"]:
        assert edge["from_id"] == "problem_root"
        assert edge["type"] == "generated_from"


@patch("src.agents.diagnostic.OpenAI")
def test_refines_on_subsequent_iteration(mock_openai_cls):
    refined = json.dumps([
        {"id": "h4", "text": "Disk pressure causing pod eviction", "domain": "infrastructure"},
        {"id": "h5", "text": "Ephemeral storage limit exceeded", "domain": "infrastructure"},
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(refined)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [
            {"id": "h1", "text": "Network issue", "status": "rejected", "domain": "network", "iteration": 0},
        ],
        "evaluations": [{"hypothesis_id": "h1", "confirmed": False}],
        "evidence": {},
        "nodes": [],
        "edges": [],
        "iteration": 1,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = generate_hypotheses(state)

    # Existing rejected + 2 new
    assert len(result["hypotheses"]) == 3
    assert result["iteration"] == 2
