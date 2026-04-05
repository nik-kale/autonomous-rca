"""Tests for the Evaluation Agent — evidence assessment and dependency detection.

All tests mock the OpenAI client so they run without an API key.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.evaluation import evaluate_evidence
from src.state import CONTRADICTS, DEPENDS_ON, REJECTED, ROOT_CAUSE


_MOCK_EVALUATION = json.dumps([
    {
        "hypothesis_id": "h1",
        "confirmed": False,
        "cause_of": None,
        "confidence": 0.2,
        "reasoning": "Network logs show all interfaces up with no issues",
    },
    {
        "hypothesis_id": "h2",
        "confirmed": True,
        "cause_of": None,
        "confidence": 0.9,
        "reasoning": "Pod eviction log directly explains the service outage",
    },
])


def _mock_openai_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


@patch("src.agents.evaluation.OpenAI")
def test_rejects_unsupported_hypothesis(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_EVALUATION)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [
            {"id": "h1", "text": "Network issue", "status": "active", "domain": "network", "iteration": 0},
            {"id": "h2", "text": "Pod eviction", "status": "active", "domain": "infrastructure", "iteration": 0},
        ],
        "evidence": {
            "h1": [{"text": "All interfaces up"}],
            "h2": [{"text": "Pod evicted due to resource limits"}],
        },
        "evaluations": [],
        "nodes": [],
        "edges": [],
        "iteration": 1,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = evaluate_evidence(state)

    rejected = [h for h in result["hypotheses"] if h["status"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["id"] == "h1"


@patch("src.agents.evaluation.OpenAI")
def test_identifies_root_cause(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_EVALUATION)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [
            {"id": "h1", "text": "Network issue", "status": "active", "domain": "network", "iteration": 0},
            {"id": "h2", "text": "Pod eviction", "status": "active", "domain": "infrastructure", "iteration": 0},
        ],
        "evidence": {
            "h1": [{"text": "All interfaces up"}],
            "h2": [{"text": "Pod evicted due to resource limits"}],
        },
        "evaluations": [],
        "nodes": [],
        "edges": [],
        "iteration": 1,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = evaluate_evidence(state)

    assert result["converged"] is True
    assert result["root_cause"] is not None

    rc_nodes = [n for n in result["nodes"] if n["type"] == ROOT_CAUSE]
    assert len(rc_nodes) == 1


@patch("src.agents.evaluation.OpenAI")
def test_emits_dependency_edges(mock_openai_cls):
    evaluation_with_deps = json.dumps([
        {
            "hypothesis_id": "h1",
            "confirmed": True,
            "cause_of": "h2",
            "confidence": 0.85,
            "reasoning": "Disk pressure caused the pod eviction",
        },
        {
            "hypothesis_id": "h2",
            "confirmed": True,
            "cause_of": None,
            "confidence": 0.7,
            "reasoning": "Pod was evicted but this is a symptom, not root cause",
        },
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(evaluation_with_deps)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [
            {"id": "h1", "text": "Disk pressure", "status": "active", "domain": "infrastructure", "iteration": 0},
            {"id": "h2", "text": "Pod eviction", "status": "active", "domain": "infrastructure", "iteration": 0},
        ],
        "evidence": {
            "h1": [{"text": "DiskPressure=True"}],
            "h2": [{"text": "Pod evicted"}],
        },
        "evaluations": [],
        "nodes": [],
        "edges": [],
        "iteration": 1,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = evaluate_evidence(state)

    dep_edges = [e for e in result["edges"] if e["type"] == DEPENDS_ON]
    assert len(dep_edges) == 1
    assert dep_edges[0]["from_id"] == "h1"
    assert dep_edges[0]["to_id"] == "h2"


@patch("src.agents.evaluation.OpenAI")
def test_emits_rejection_nodes(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(_MOCK_EVALUATION)
    mock_openai_cls.return_value = mock_client

    state = {
        "problem_statement": "Service down",
        "hypotheses": [
            {"id": "h1", "text": "Network issue", "status": "active", "domain": "network", "iteration": 0},
            {"id": "h2", "text": "Pod eviction", "status": "active", "domain": "infrastructure", "iteration": 0},
        ],
        "evidence": {
            "h1": [{"text": "All interfaces up"}],
            "h2": [{"text": "Pod evicted"}],
        },
        "evaluations": [],
        "nodes": [],
        "edges": [],
        "iteration": 1,
        "max_iterations": 5,
        "converged": False,
        "root_cause": None,
    }

    result = evaluate_evidence(state)

    rej_nodes = [n for n in result["nodes"] if n["type"] == REJECTED]
    assert len(rej_nodes) == 1

    contra_edges = [e for e in result["edges"] if e["type"] == CONTRADICTS]
    assert len(contra_edges) == 1
