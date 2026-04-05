"""Investigation state definition for the agentic RCA pipeline.

This module defines the core data structure that flows through every node in
the LangGraph investigation graph.  The two key design choices are:

1. **Typed state** — ``InvestigationState`` is a ``TypedDict`` so every agent
   node gets autocompletion and type-checking for the fields it reads/writes.

2. **Accumulating reducers** — The ``nodes`` and ``edges`` lists use
   ``Annotated[list, operator.add]`` so that each agent can *return* new graph
   fragments and LangGraph will *append* them to the running lists rather than
   replacing them.  At the end of an investigation, these lists contain the
   complete Investigation Graph — the structured reasoning trace.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

# ---------------------------------------------------------------------------
# Node types — vertices in the Investigation Graph
# ---------------------------------------------------------------------------
PROBLEM = "problem"
HYPOTHESIS = "hypothesis"
EVIDENCE = "evidence"
REJECTED = "rejected"
ROOT_CAUSE = "root_cause"

# ---------------------------------------------------------------------------
# Edge types — directed relationships between nodes
# ---------------------------------------------------------------------------
GENERATED_FROM = "generated_from"
SUPPORTS = "supports"
CONTRADICTS = "contradicts"
CAUSES = "causes"
DEPENDS_ON = "depends_on"


class InvestigationState(TypedDict):
    """State that flows through the LangGraph investigation pipeline.

    Fields without a reducer are *replaced* by the returning node's value.
    Fields annotated with ``operator.add`` *accumulate* across every node
    invocation — this is how the Investigation Graph grows iteration by
    iteration.
    """

    problem_statement: str

    # Hypotheses generated (and refined) by the Diagnostic Agent
    hypotheses: list[dict]  # [{id, text, status, domain, iteration}]

    # Evidence retrieved for each hypothesis, keyed by hypothesis id
    evidence: dict  # {hypothesis_id: [evidence_items]}

    # Evaluation results from the Evaluation Agent
    evaluations: list[dict]  # [{hypothesis_id, confirmed, cause_of, confidence, reasoning}]

    # --- Investigation Graph (accumulated via reducer) ---
    nodes: Annotated[list[dict], operator.add]
    # [{id, type, text, iteration, confidence}]

    edges: Annotated[list[dict], operator.add]
    # [{from_id, to_id, type, iteration}]

    # Loop control
    iteration: int
    max_iterations: int
    converged: bool
    root_cause: Optional[str]
