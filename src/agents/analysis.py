"""Analysis Agent — investigation synthesis.

This is the terminal node in the investigation loop.  It receives the full
investigation state (including the accumulated Investigation Graph) and
produces a final summary: the root cause, the reasoning path, recommended
resolution steps, and an overall confidence score.

If the evaluation loop already identified a root cause with high confidence,
this agent enriches the explanation.  If the loop timed out or failed to
converge, this agent synthesizes the best available evidence into a
best-effort conclusion.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm import llm_call_json
from src.state import ROOT_CAUSE, CAUSES, InvestigationState

_SYSTEM_PROMPT = """\
You are an expert site-reliability engineer writing a root-cause analysis
report.  You are given:

- The original problem statement
- A list of hypotheses (some confirmed, some rejected)
- Evaluation results with causal dependency information
- The Investigation Graph (nodes and edges)

Produce a JSON object with:
{{
  "root_cause": "<concise root-cause statement>",
  "reasoning_path": ["<step 1>", "<step 2>", ...],
  "resolution_steps": ["<action 1>", "<action 2>", ...],
  "confidence": 0.0-1.0,
  "summary": "<2-3 sentence executive summary>"
}}

Return ONLY the JSON object — no markdown fences.
"""


def synthesize_investigation(state: InvestigationState) -> dict[str, Any]:
    """Synthesize the investigation into a final root-cause report.

    Returns a partial state update containing:
    - ``root_cause`` — final root-cause statement (may refine the evaluator's)
    - ``nodes`` — a synthesis node for the Investigation Graph
    - ``edges`` — edge from the confirmed hypothesis to the synthesis node
    """
    user_parts = [
        f"Problem: {state['problem_statement']}",
        f"\nHypotheses:\n{json.dumps(state.get('hypotheses', []), indent=2)}",
        f"\nEvaluations:\n{json.dumps(state.get('evaluations', []), indent=2)}",
        f"\nInvestigation Graph Nodes ({len(state.get('nodes', []))}):",
        json.dumps(state.get("nodes", [])[:30], indent=2),
        f"\nInvestigation Graph Edges ({len(state.get('edges', []))}):",
        json.dumps(state.get("edges", [])[:30], indent=2),
    ]
    user_msg = "\n".join(user_parts)

    synthesis: dict = llm_call_json(_SYSTEM_PROMPT, user_msg)

    root_cause = synthesis.get("root_cause", state.get("root_cause", "Undetermined"))
    iteration = state.get("iteration", 0)

    new_nodes = [
        {
            "id": "synthesis",
            "type": ROOT_CAUSE,
            "text": root_cause,
            "iteration": iteration,
            "confidence": synthesis.get("confidence", 0.0),
        }
    ]

    new_edges = []
    for h in state.get("hypotheses", []):
        if h.get("status") == "root_cause":
            new_edges.append(
                {
                    "from_id": h["id"],
                    "to_id": "synthesis",
                    "type": CAUSES,
                    "iteration": iteration,
                }
            )

    return {
        "root_cause": root_cause,
        "nodes": new_nodes,
        "edges": new_edges,
        "converged": True,
    }
