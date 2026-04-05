# Architecture

This document explains the system architecture of the agentic root-cause
analysis pipeline.

## Overview

The system implements a **cyclic multi-agent architecture** using LangGraph's
`StateGraph`.  Four specialized agents operate on a shared typed state,
iterating until convergence.

```
┌─────────────────────────────────────────────────────┐
│                  InvestigationState                  │
│                                                     │
│  problem_statement    hypotheses    evidence         │
│  evaluations          nodes ⊕      edges ⊕          │
│  iteration            converged    root_cause        │
│                                                     │
│  ⊕ = Annotated[list, operator.add]  (accumulates)   │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Diagnose │  │ Retrieve │  │ Evaluate │
    │  Agent   │──│  Agent   │──│  Agent   │
    └──────────┘  └──────────┘  └────┬─────┘
                                     │
                              ┌──────┴──────┐
                              │should_continue│
                              └──────┬──────┘
                             ╱               ╲
                     "diagnose"           "analyze"
                         │                    │
                    (loop back)          ┌────▼────┐
                                         │Analysis │
                                         │  Agent  │
                                         └────┬────┘
                                              │
                                             END
```

## State Design

The `InvestigationState` TypedDict uses two critical patterns:

### Plain fields (replaced each iteration)
Fields like `hypotheses`, `evidence`, `evaluations`, `iteration`, and
`converged` are **replaced** by each agent's return value.  The latest
version wins.

### Reducer fields (accumulated across iterations)
The `nodes` and `edges` fields use `Annotated[list[dict], operator.add]`.
When an agent returns new nodes or edges, LangGraph **appends** them to the
existing lists instead of replacing them.  This is how the Investigation
Graph grows iteration by iteration without any agent needing to manage the
full history.

## Agent Contract

Every agent follows the same contract:

```python
def agent_function(state: InvestigationState) -> dict:
    # Read what you need from state
    # Do your work (LLM call, retrieval, etc.)
    # Return ONLY the fields you're updating
    return {"hypotheses": [...], "nodes": [...], "edges": [...]}
```

The partial return is the key: agents are decoupled from each other and
from the full state shape.  This makes them independently testable.

## Convergence Detection

The `should_continue` function implements three convergence conditions:

1. **Explicit convergence** — the Evaluation Agent sets `converged=True`
   when a hypothesis reaches high confidence as a root cause.
2. **Budget exhaustion** — `iteration >= max_iterations` prevents runaway
   loops.
3. **Resolution completeness** — if every hypothesis is either `rejected`
   or `root_cause`, there is nothing left to investigate.

## Retrieval Strategy Selection

The Retrieval Agent delegates to a strategy selector that chooses the
best search method per hypothesis:

| Signal | Strategy | Rationale |
|--------|----------|-----------|
| Error codes, status codes | Keyword | Exact match is fastest |
| Large log corpus (>10 items) | TF-IDF | Rare-term weighting surfaces anomalies |
| Fuzzy/conceptual query | Semantic | Embedding similarity handles synonyms |

In production, each strategy would call a different backend: keyword maps
to Elasticsearch `match_phrase`, TF-IDF maps to `more_like_this`, and
semantic maps to a vector database query.

## Extension Points

To adapt this pattern for real infrastructure:

1. **Replace mock data** — swap `src/evidence/mock_data.py` with connectors
   to your log store, metrics API, and configuration management system.
2. **Add retrieval strategies** — implement structured query for databases,
   graph traversal for dependency maps, etc.
3. **Tune prompts** — the generic SRE prompts work for demonstration but
   should be refined for your specific domain and failure modes.
4. **Add persistence** — LangGraph supports checkpointing; add a checkpointer
   to resume interrupted investigations.
5. **Add human-in-the-loop** — use LangGraph's interrupt mechanism to let
   an operator confirm high-impact conclusions before they are finalized.
