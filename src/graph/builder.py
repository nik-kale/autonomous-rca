"""LangGraph StateGraph construction for the investigation pipeline.

This module wires the four agent nodes into a cyclic state machine:

    diagnose → retrieve → evaluate → (conditional) → diagnose OR analyze → END

The conditional edge implements convergence detection: the investigation loops
until the Evaluation Agent signals convergence, the iteration budget is
exhausted, or no hypothesis can be further refined.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.analysis import synthesize_investigation
from src.agents.diagnostic import generate_hypotheses
from src.agents.evaluation import evaluate_evidence
from src.agents.retrieval import retrieve_evidence
from src.state import InvestigationState


def should_continue(state: InvestigationState) -> Literal["diagnose", "analyze"]:
    """Determine whether to iterate or converge.

    Returns ``"diagnose"`` to continue investigating, or ``"analyze"`` to
    proceed to the final synthesis step.
    """
    if state.get("converged"):
        return "analyze"

    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "analyze"

    # If every hypothesis has been resolved, there's nothing left to refine
    refinable = [
        h
        for h in state.get("hypotheses", [])
        if h.get("status") not in ("rejected", "root_cause")
    ]
    if not refinable:
        return "analyze"

    return "diagnose"


def build_investigation_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph investigation pipeline.

    Returns a compiled ``CompiledStateGraph`` ready for ``.invoke()`` or
    ``.stream()`` calls.
    """
    workflow = StateGraph(InvestigationState)

    workflow.add_node("diagnose", generate_hypotheses)
    workflow.add_node("retrieve", retrieve_evidence)
    workflow.add_node("evaluate", evaluate_evidence)
    workflow.add_node("analyze", synthesize_investigation)

    workflow.set_entry_point("diagnose")

    workflow.add_edge("diagnose", "retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {"diagnose": "diagnose", "analyze": "analyze"},
    )
    workflow.add_edge("analyze", END)

    return workflow.compile()
