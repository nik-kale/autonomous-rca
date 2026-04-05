"""Evaluation Agent — evidence assessment and dependency detection.

For each active hypothesis the agent asks the LLM to:

1. Determine whether the retrieved evidence **supports**, **contradicts**,
   or is **inconclusive** for the hypothesis.
2. Detect **causal dependencies** between hypotheses — the ``cause_of``
   field.  When hypothesis A is the cause of hypothesis B, we emit a
   ``depends_on`` edge so the search space can be pruned (if A is the root
   cause, B is a symptom and need not be investigated further).

The ``cause_of`` field is the critical architectural innovation: it turns a
flat list of parallel hypotheses into a **directed causal chain**, enabling
the Investigation Graph to capture the reasoning path from symptoms to root
cause.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from src.state import (
    CAUSES,
    CONTRADICTS,
    DEPENDS_ON,
    REJECTED,
    ROOT_CAUSE,
    SUPPORTS,
    InvestigationState,
)

_SYSTEM_PROMPT = """\
You are an expert site-reliability engineer evaluating root-cause hypotheses.

For each hypothesis and its supporting evidence, determine:
1. Whether the evidence confirms or contradicts the hypothesis.
2. Whether this hypothesis is the *cause of* another hypothesis (causal
   dependency).  If hypothesis A caused the symptoms described by hypothesis B,
   set cause_of to the id of B.
3. A confidence score (0.0 to 1.0).

Return ONLY a JSON array — no markdown fences:
[
  {
    "hypothesis_id": "<id>",
    "confirmed": true or false,
    "cause_of": "<id of dependent hypothesis, or null>",
    "confidence": 0.85,
    "reasoning": "<one sentence explaining the judgement>"
  }
]
"""


def _llm_call(system: str, user: str) -> str:
    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def evaluate_evidence(state: InvestigationState) -> dict[str, Any]:
    """Evaluate evidence against hypotheses; detect causal dependencies.

    Returns a partial state update containing:
    - ``evaluations`` — list of structured evaluation dicts
    - ``hypotheses`` — updated hypothesis list with status changes
    - ``nodes`` — graph nodes for rejections and root-cause findings
    - ``edges`` — ``supports``, ``contradicts``, ``causes``, ``depends_on``
    - ``converged`` — True when a root cause is identified with high confidence
    """
    hypotheses = list(state.get("hypotheses", []))
    evidence = state.get("evidence", {})

    active = [h for h in hypotheses if h.get("status") == "active"]
    if not active:
        return {"converged": True}

    # Build per-hypothesis evidence summary for the prompt
    evidence_text_parts: list[str] = []
    for h in active:
        items = evidence.get(h["id"], [])
        texts = [item.get("text", "") for item in items[:5]] if isinstance(items, list) else []
        evidence_text_parts.append(
            f"Hypothesis {h['id']} ({h['text']}):\n" + "\n".join(f"  - {t}" for t in texts)
        )

    user_msg = (
        f"Problem: {state['problem_statement']}\n\n"
        + "\n\n".join(evidence_text_parts)
    )

    raw = _llm_call(_SYSTEM_PROMPT, user_msg)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]

    evaluations: list[dict] = json.loads(cleaned)

    new_nodes: list[dict] = []
    new_edges: list[dict] = []
    converged = False
    root_cause_text: str | None = None

    h_lookup = {h["id"]: h for h in hypotheses}

    for ev in evaluations:
        h_id = ev.get("hypothesis_id", "")
        confirmed = ev.get("confirmed", False)
        cause_of = ev.get("cause_of")
        confidence = ev.get("confidence", 0.0)
        iteration = state.get("iteration", 0)

        if h_id in h_lookup:
            if confirmed:
                # High-confidence confirmed hypothesis with no further cause
                if confidence >= 0.8 and not cause_of:
                    h_lookup[h_id]["status"] = "root_cause"
                    converged = True
                    root_cause_text = (
                        f"{h_lookup[h_id]['text']} "
                        f"(confidence: {confidence:.0%}, "
                        f"reasoning: {ev.get('reasoning', '')})"
                    )
                    new_nodes.append(
                        {
                            "id": f"rc_{h_id}",
                            "type": ROOT_CAUSE,
                            "text": h_lookup[h_id]["text"],
                            "iteration": iteration,
                            "confidence": confidence,
                        }
                    )
                    new_edges.append(
                        {
                            "from_id": h_id,
                            "to_id": f"rc_{h_id}",
                            "type": CAUSES,
                            "iteration": iteration,
                        }
                    )
                else:
                    new_edges.append(
                        {
                            "from_id": h_id,
                            "to_id": h_id,
                            "type": SUPPORTS,
                            "iteration": iteration,
                        }
                    )
            else:
                h_lookup[h_id]["status"] = "rejected"
                new_nodes.append(
                    {
                        "id": f"rej_{h_id}",
                        "type": REJECTED,
                        "text": f"Rejected: {h_lookup[h_id]['text']}",
                        "iteration": iteration,
                        "confidence": confidence,
                    }
                )
                new_edges.append(
                    {
                        "from_id": h_id,
                        "to_id": f"rej_{h_id}",
                        "type": CONTRADICTS,
                        "iteration": iteration,
                    }
                )

            # Dependency pruning — the key innovation
            if cause_of and cause_of in h_lookup:
                new_edges.append(
                    {
                        "from_id": h_id,
                        "to_id": cause_of,
                        "type": DEPENDS_ON,
                        "iteration": iteration,
                    }
                )

    updated_hypotheses = [h_lookup.get(h["id"], h) for h in hypotheses]

    result: dict[str, Any] = {
        "evaluations": evaluations,
        "hypotheses": updated_hypotheses,
        "nodes": new_nodes,
        "edges": new_edges,
        "converged": converged,
    }
    if root_cause_text:
        result["root_cause"] = root_cause_text

    return result
