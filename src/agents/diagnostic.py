"""Diagnostic Agent — hypothesis generation.

This node takes the current investigation state and produces 2-3 hypotheses
about potential root causes.  On the first iteration it generates broad
hypotheses spanning different domains (network, application, infrastructure).
On subsequent iterations it refines hypotheses based on evaluation feedback
from the previous cycle.

The agent emits Investigation Graph fragments: a ``hypothesis`` node for each
new hypothesis and a ``generated_from`` edge linking it to the problem node
(or to the evidence that inspired the refinement).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from src.llm import llm_call_json
from src.state import (
    GENERATED_FROM,
    HYPOTHESIS,
    PROBLEM,
    InvestigationState,
)

_SYSTEM_PROMPT = """\
You are an expert site-reliability engineer performing root-cause analysis.

Given a problem statement and (optionally) prior evaluation results, generate
{num_hypotheses} hypotheses that could explain the root cause.  Each hypothesis
should target a different technical domain when possible.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{
    "id": "<short unique id like h1, h2, ...>",
    "text": "<one-sentence hypothesis>",
    "domain": "<one of: network, application, infrastructure, database, security>"
  }}
]
"""

_REFINEMENT_ADDENDUM = """
Previous evaluation results (use these to refine your hypotheses):
{evaluations}

Do NOT regenerate hypotheses that have already been rejected.  Focus on
refining or deepening the remaining lines of inquiry.
"""


def generate_hypotheses(state: InvestigationState) -> dict[str, Any]:
    """Generate or refine root-cause hypotheses.

    Returns a partial state update containing:
    - ``hypotheses`` — the full (replaced) hypothesis list
    - ``nodes`` — new Investigation Graph nodes (appended via reducer)
    - ``edges`` — new Investigation Graph edges (appended via reducer)
    """
    iteration = state.get("iteration", 0)
    existing_hypotheses = state.get("hypotheses", [])
    evaluations = state.get("evaluations", [])

    num_hypotheses = 3 if iteration == 0 else 2
    system = _SYSTEM_PROMPT.format(num_hypotheses=num_hypotheses)

    if iteration > 0 and evaluations:
        system += _REFINEMENT_ADDENDUM.format(
            evaluations=json.dumps(evaluations, indent=2)
        )

    user_msg = f"Problem: {state['problem_statement']}"
    if state.get("evidence"):
        evidence_summary = {
            k: [e["text"] for e in v[:3]]
            for k, v in state["evidence"].items()
            if isinstance(v, list)
        }
        user_msg += f"\n\nAvailable evidence (sample):\n{json.dumps(evidence_summary, indent=2)}"

    new_hypotheses_raw: list[dict] = llm_call_json(system, user_msg, temperature=0.3)

    existing_ids = {h["id"] for h in existing_hypotheses}
    new_nodes: list[dict] = []
    new_edges: list[dict] = []

    for h in new_hypotheses_raw:
        raw_id = h.get("id", f"h{uuid.uuid4().hex[:6]}")
        h_id = f"i{iteration}_{raw_id}"
        while h_id in existing_ids:
            h_id = f"i{iteration}_{raw_id}_{uuid.uuid4().hex[:4]}"
        existing_ids.add(h_id)

        h["id"] = h_id
        h["status"] = "active"
        h["iteration"] = iteration

        new_nodes.append(
            {
                "id": h_id,
                "type": HYPOTHESIS,
                "text": h["text"],
                "iteration": iteration,
                "confidence": None,
            }
        )
        new_edges.append(
            {
                "from_id": "problem_root",
                "to_id": h_id,
                "type": GENERATED_FROM,
                "iteration": iteration,
            }
        )

    if iteration == 0:
        new_nodes.insert(
            0,
            {
                "id": "problem_root",
                "type": PROBLEM,
                "text": state["problem_statement"],
                "iteration": 0,
                "confidence": None,
            },
        )

    all_hypotheses = existing_hypotheses + new_hypotheses_raw

    return {
        "hypotheses": all_hypotheses,
        "nodes": new_nodes,
        "edges": new_edges,
        "iteration": iteration + 1,
    }
