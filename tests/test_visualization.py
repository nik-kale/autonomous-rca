"""Tests for the Investigation Graph visualization and analysis utilities."""

from __future__ import annotations

import pytest

from src.graph.visualization import (
    export_graph_json,
    extract_root_cause_path,
)
from src.state import CAUSES, GENERATED_FROM, PROBLEM, HYPOTHESIS, ROOT_CAUSE, SUPPORTS


_SAMPLE_NODES = [
    {"id": "problem_root", "type": PROBLEM, "text": "Service down", "iteration": 0, "confidence": None},
    {"id": "h1", "type": HYPOTHESIS, "text": "Disk pressure", "iteration": 1, "confidence": None},
    {"id": "h2", "type": HYPOTHESIS, "text": "Network failure", "iteration": 1, "confidence": None},
    {"id": "rc_h1", "type": ROOT_CAUSE, "text": "Disk pressure", "iteration": 2, "confidence": 0.92},
]

_SAMPLE_EDGES = [
    {"from_id": "problem_root", "to_id": "h1", "type": GENERATED_FROM, "iteration": 1},
    {"from_id": "problem_root", "to_id": "h2", "type": GENERATED_FROM, "iteration": 1},
    {"from_id": "h1", "to_id": "rc_h1", "type": CAUSES, "iteration": 2},
]


class TestExtractRootCausePath:
    def test_finds_path_from_problem_to_root_cause(self):
        path = extract_root_cause_path(_SAMPLE_NODES, _SAMPLE_EDGES)

        assert len(path) > 0
        assert path[0]["type"] == PROBLEM
        assert path[-1]["type"] == ROOT_CAUSE

    def test_returns_empty_when_no_root_cause(self):
        nodes = [n for n in _SAMPLE_NODES if n["type"] != ROOT_CAUSE]
        path = extract_root_cause_path(nodes, _SAMPLE_EDGES)
        assert path == []

    def test_returns_empty_when_no_problem_node(self):
        nodes = [n for n in _SAMPLE_NODES if n["type"] != PROBLEM]
        path = extract_root_cause_path(nodes, _SAMPLE_EDGES)
        assert path == []

    def test_returns_empty_for_empty_graph(self):
        path = extract_root_cause_path([], [])
        assert path == []

    def test_path_contains_intermediate_hypothesis(self):
        path = extract_root_cause_path(_SAMPLE_NODES, _SAMPLE_EDGES)

        path_types = [step["type"] for step in path]
        assert HYPOTHESIS in path_types


class TestExportGraphJson:
    def test_includes_all_nodes_and_edges(self):
        result = export_graph_json(_SAMPLE_NODES, _SAMPLE_EDGES)

        assert result["nodes"] == _SAMPLE_NODES
        assert result["edges"] == _SAMPLE_EDGES

    def test_summary_has_correct_counts(self):
        result = export_graph_json(_SAMPLE_NODES, _SAMPLE_EDGES)

        assert result["summary"]["total_nodes"] == 4
        assert result["summary"]["total_edges"] == 3

    def test_summary_has_type_distribution(self):
        result = export_graph_json(_SAMPLE_NODES, _SAMPLE_EDGES)

        assert result["summary"]["node_types"][PROBLEM] == 1
        assert result["summary"]["node_types"][HYPOTHESIS] == 2
        assert result["summary"]["node_types"][ROOT_CAUSE] == 1

        assert result["summary"]["edge_types"][GENERATED_FROM] == 2
        assert result["summary"]["edge_types"][CAUSES] == 1

    def test_handles_empty_graph(self):
        result = export_graph_json([], [])

        assert result["summary"]["total_nodes"] == 0
        assert result["summary"]["total_edges"] == 0
