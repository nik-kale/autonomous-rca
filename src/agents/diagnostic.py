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
import os
import uuid
from typing import Any

from openai import OpenAI

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


def _llm_call(system: str, user: str) -> str:
    """Make a single chat completion call and return the text response."""
    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


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

    raw = _llm_call(system, user_msg)

    # Robust JSON extraction — handle markdown fences if the model wraps them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    new_hypotheses_raw: list[dict] = json.loads(cleaned)

    # Merge with existing, tagging each with iteration + status
    new_nodes: list[dict] = []
    new_edges: list[dict] = []

    for h in new_hypotheses_raw:
        h_id = h.get("id", f"h{uuid.uuid4().hex[:6]}")
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

    # On iteration 0 add the problem node itself
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
